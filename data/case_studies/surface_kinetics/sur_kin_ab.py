import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math
import pandas as pd
import openpyxl
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import yaml

try:
    from surf_cythonized_speedup import compute_dydt as _compute_dydt_fast
except ImportError:
    _compute_dydt_fast = None

# ---------------------------------------------------------------------------
#  Embedded standalone surface-kinetics model (substrate-based CVD)
# ---------------------------------------------------------------------------
PERIODIC_TABLE = {
    "H": 1.008,
    "He": 4.0026,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "Ar": 39.948,
}


def _strip_surface_suffix(name):
    for suf in ("_s", "_NT", "_g"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def parse_formula(formula):
    token_re = re.compile(r"([A-Z][a-z]?)(\d*)")
    out: Dict[str, int] = {}
    for el, num in token_re.findall(formula):
        out[el] = out.get(el, 0) + int(num or "1")
    return out


def molecular_weight(formula, atomic_masses):
    base = _strip_surface_suffix(formula)
    comp = parse_formula(base)
    am = atomic_masses or {}
    mw = 0.0
    for el, cnt in comp.items():
        mass = am.get(el, PERIODIC_TABLE.get(el))
        mw += float(mass) * float(cnt)
    return float(mw)


def catalyst_area_fraction_A_star(*, rho_part_m2, d_np_outer_m):
    rho = float(rho_part_m2)
    d = float(d_np_outer_m)
    if rho <= 0.0 or d <= 0.0:
        return 0.0
    a_star = rho * (math.pi / 4.0) * d * d
    return float(min(1.0, max(0.0, a_star)))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


class SurfaceChemistryLoader:
    def __init__(self, yaml_file):
        self.yaml_file = yaml_file

    @staticmethod
    def _parse_rate_constant_dict(d: Any):
        if not isinstance(d, dict):
            return (0.0, 0.0, 0.0)
        return (
            _safe_float(d.get("A", 0.0), 0.0),
            _safe_float(d.get("b", 0.0), 0.0),
            _safe_float(d.get("Ea", 0.0), 0.0),
        )

    @staticmethod
    def _parse_equation_side(side) -> Tuple[Dict[str, float], float]:
        counts: Dict[str, float] = {}
        m_order = 0.0
        if not side or not side.strip():
            return counts, m_order
        side = side.split("#", 1)[0]
        side = side.replace("+", " + ")
        parts = [p.strip() for p in side.split("+") if p.strip()]
        token_re = re.compile(r"^\s*(?:(\d+(?:\.\d+)?)\s*)?([A-Za-z0-9_()]+)\s*$")
        for p in parts:
            m = token_re.match(p)
            if not m:
                raise ValueError(f"Could not parse reaction token '{p}' in '{side}'")
            coeff = float(m.group(1) or 1.0)
            sp = m.group(2)
            if sp == "M":
                m_order += coeff
                continue
            counts[sp] = counts.get(sp, 0.0) + coeff
        return counts, m_order

    def _extract_reaction_blocks(self, raw):
        lines = raw.splitlines()
        out: List[Dict[str, Any]] = []
        cur: Optional[Dict[str, Any]] = None
        i = 0

        def flush():
            nonlocal cur
            if cur is not None:
                out.append(cur)
                cur = None

        while i < len(lines):
            line = lines[i]
            if re.match(r"^\s*-\s*equation\s*:", line):
                flush()
                eq = line.split(":", 1)[1].strip()
                cur = {"equation": eq, "rate-constant": {}, "impingements": None, "efficiencies": {}}
                i += 1
                continue

            if cur is None:
                i += 1
                continue

            if "rate-constant" in line:
                rhs = line.split(":", 1)[1].strip()
                if rhs.lower().startswith("impingements"):
                    cur["impingements"] = {"sticking": 1.0}
                    cur["rate-constant"] = {}
                    i += 1
                    continue
                if rhs.startswith("{") and rhs.endswith("}"):
                    inner = rhs[1:-1]
                    kdict: Dict[str, float] = {}
                    for item in inner.split(","):
                        if ":" not in item:
                            continue
                        k, v = item.split(":", 1)
                        try:
                            kdict[k.strip()] = float(v.strip())
                        except Exception:
                            pass
                    cur["rate-constant"] = kdict
                else:
                    block = [line]
                    j = i + 1
                    while j < len(lines) and not re.match(r"^\s*-\s*equation\s*:", lines[j]):
                        if lines[j].strip() == "":
                            j += 1
                            continue
                        block.append(lines[j])
                        j += 1
                    try:
                        parsed = yaml.safe_load("\n".join(block))
                        if isinstance(parsed, dict) and "rate-constant" in parsed:
                            cur["rate-constant"] = parsed["rate-constant"]
                    except Exception:
                        pass
                i += 1
                continue

            if re.search(r"\bimpingements\b", line):
                rhs = line.split(":", 1)[1].strip()
                if rhs.lower() in ("true", "yes", "on"):
                    cur["impingements"] = {"sticking": 1.0}
                elif rhs.lower() in ("false", "no", "off", "null", "~", ""):
                    cur["impingements"] = None
                elif rhs.startswith("{") and rhs.endswith("}"):
                    inner = rhs[1:-1]
                    dd: Dict[str, Any] = {}
                    for item in inner.split(","):
                        if ":" not in item:
                            continue
                        k, v = item.split(":", 1)
                        vv = v.strip()
                        try:
                            dd[k.strip()] = float(vv)
                        except Exception:
                            dd[k.strip()] = vv
                    cur["impingements"] = dd
                else:
                    try:
                        cur["impingements"] = yaml.safe_load(rhs)
                    except Exception:
                        cur["impingements"] = {"sticking": 1.0}
                i += 1
                continue

            if re.match(r"^\s*efficiencies\s*:", line):
                rhs = line.split(":", 1)[1].strip()
                eff: Dict[str, float] = {}
                if rhs.startswith("{") and rhs.endswith("}"):
                    inner = rhs[1:-1]
                    for item in inner.split(","):
                        if ":" not in item:
                            continue
                        k, v = item.split(":", 1)
                        try:
                            eff[k.strip()] = float(v.strip())
                        except Exception:
                            pass
                    cur["efficiencies"] = eff
                    i += 1
                    continue
                j = i + 1
                while j < len(lines):
                    nxt = lines[j]
                    if re.match(r"^\s*-\s*equation\s*:", nxt) or re.match(r"^\s*\w+\s*:", nxt) and not nxt.startswith(" "):
                        break
                    m = re.match(r"^\s*([A-Za-z0-9_()]+)\s*:\s*([0-9eE\+\-\.]+)\s*$", nxt)
                    if m:
                        try:
                            eff[m.group(1)] = float(m.group(2))
                        except Exception:
                            pass
                    j += 1
                cur["efficiencies"] = eff
                i = j
                continue

            i += 1

        flush()
        return out

    @staticmethod
    def _extract_species_list(data, raw):
        if isinstance(data, dict):
            phases = data.get("phases")
            if isinstance(phases, dict) and isinstance(phases.get("species"), list):
                return [str(s) for s in phases["species"]]
            if isinstance(phases, list):
                for ph in phases:
                    if isinstance(ph, dict) and isinstance(ph.get("species"), list):
                        return [str(s) for s in ph["species"]]

        species: set[str] = set()
        eq_re = re.compile(r"^\s*-\s*equation\s*:\s*(.+?)\s*(?:#.*)?$", re.MULTILINE)
        for m in eq_re.finditer(raw):
            eq = m.group(1).strip()
            if "=>" not in eq or eq.upper() == "N/A":
                continue
            lhs, rhs = eq.split("=>", 1)
            for side in (lhs, rhs):
                side = side.split("#", 1)[0].replace("+", " + ")
                for tok in side.split("+"):
                    tok = tok.strip()
                    if not tok:
                        continue
                    tok = re.sub(r"^\d+(?:\.\d+)?\s*", "", tok)
                    if tok == "M":
                        continue
                    species.add(tok)
        return sorted(species)

    def load(self) -> Dict[str, Any]:
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            raw = f.read()
        raw_clean = re.sub(r"(equation|rate-constant)\s*:(?=\S)", r"\1: ", raw)
        try:
            data = yaml.safe_load(raw_clean)
        except Exception:
            data = None

        reactions = self._extract_reaction_blocks(raw_clean)
        species = self._extract_species_list(data, raw_clean)
        atomic_masses: Dict[str, float] = {}
        if isinstance(data, dict):
            phases = data.get("phases")
            if isinstance(phases, dict) and isinstance(phases.get("atomic_masses"), dict):
                atomic_masses = {str(k): float(v) for k, v in phases["atomic_masses"].items() if isinstance(v, (int, float))}
            if isinstance(phases, list):
                for ph in phases:
                    if isinstance(ph, dict) and isinstance(ph.get("atomic_masses"), dict):
                        atomic_masses = {str(k): float(v) for k, v in ph["atomic_masses"].items() if isinstance(v, (int, float))}
                        break

        if not reactions:
            raise ValueError(f"No reactions found in {self.yaml_file}")
        return {"species": species, "atomic_masses": atomic_masses, "reaction_entries": reactions}


@dataclass
class ReactionPair:
    equation_fwd: str
    reactants: Dict[str, float]
    products: Dict[str, float]
    m_order_fwd: float
    m_order_rev: float
    eff_fwd: Dict[str, float]
    eff_rev: Dict[str, float]
    kf: Tuple[float, float, float]  # (A, b, Ea[J/mol])
    kr: Tuple[float, float, float]  # (A, b, Ea[J/mol])
    has_reverse: bool
    impingement: Optional[Dict[str, Any]]

    @property
    def net_stoich(self) -> Dict[str, float]:
        nu: Dict[str, float] = {}
        for s, c in self.products.items():
            nu[s] = nu.get(s, 0.0) + c
        for s, c in self.reactants.items():
            nu[s] = nu.get(s, 0.0) - c
        nu.pop("M", None)
        return nu


class ParametricSurfaceReactorSimulation:
    def __init__(
        self,
        yaml_file: str,
        *,
        rho_s0_sites_m2: float = 5.0e15,
        a_cat_m2_m3: float = 1.0,
        impingement_model: str = "paper",
        A_star: float = 1.0,
        inert_species: Optional[List[str]] = None,
        constant_species: Optional[Dict[str, float]] = None,
    ):
        self.yaml_file = str(yaml_file)
        self.R = 8.314462618
        self.NA = 6.02214076e23
        self.kB = 1.380649e-23
        self.rho_s0 = float(rho_s0_sites_m2)
        self.a_cat = float(a_cat_m2_m3)
        self.impingement_model = str(impingement_model).strip().lower()
        if self.impingement_model not in ("heuristic", "paper"):
            raise ValueError("impingement_model must be 'heuristic' or 'paper'")
        self.A_star = float(A_star)
        if not (0.0 <= self.A_star <= 1.0):
            raise ValueError("A_star must be between 0 and 1")
        self.inert_species = inert_species or ["N2", "Ar", "He"]
        self.constant_species = dict(constant_species or {"NT": 1.0})
        self._load_mechanism()

        self._mw_by_species = {}
        for s in self.species_all:
            if s == "S" or s == "M":
                continue
            try:
                self._mw_by_species[s] = molecular_weight(s, self.atomic_masses)
            except Exception:
                pass

        self.surface_species = sorted([s for s in self.species_all if self._is_surface_species(s) and s != "S"])
        self.gas_species = sorted(
            [
                s
                for s in self.species_all
                if (not self._is_surface_species(s)) and s not in ("M",) and (not s.endswith("_NT")) and (s not in self.constant_species)
            ]
        )

        self.surface_index = {s: i for i, s in enumerate(self.surface_species)}
        self.gas_index = {s: i for i, s in enumerate(self.gas_species)}

        self._rxn_reactants = []
        self._rxn_products = []
        self._rxn_nu_gas = []
        self._rxn_nu_surf = []
        for rxn in self.reactions:
            self._rxn_reactants.append(list(rxn.reactants.items()))
            self._rxn_products.append(list(rxn.products.items()))
            nu = rxn.net_stoich
            self._rxn_nu_gas.append({s: v for s, v in nu.items() if s in self.gas_index})
            self._rxn_nu_surf.append({s: v for s, v in nu.items() if s in self.surface_index})

    @staticmethod
    def _is_surface_species(name: str) -> bool:
        return name == "S" or name.endswith("_s")

    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    @staticmethod
    def _arrhenius_to_si(k_tuple: Tuple[float, float, float], reaction_order: float) -> Tuple[float, float, float]:
        A, b, Ea_cal_mol = k_tuple
        # Ea: cal/mol -> J/mol
        Ea = float(Ea_cal_mol) * 4.184
        # A: cm-based -> m-based, ((cm^3/mol)^(order-1)/s) -> ((m^3/mol)^(order-1)/s)
        if reaction_order > 1.0:
            A = float(A) * ((1e-6) ** (reaction_order - 1.0))
        return float(A), float(b), float(Ea)

    @staticmethod
    def _parse_equation_side(side: str) -> Dict[str, float]:
        counts: Dict[str, float] = {}
        if not side:
            return counts
        side = side.split("#", 1)[0].strip()
        if not side:
            return counts
        pieces = [p.strip() for p in side.replace("+", " + ").split("+") if p.strip()]
        tok_re = re.compile(r"^\s*(?:(\d+(?:\.\d+)?)\s*)?([A-Za-z0-9_()]+)\s*$")
        for p in pieces:
            m = tok_re.match(p)
            if not m:
                continue
            coeff = float(m.group(1) or 1.0)
            sp = m.group(2)
            counts[sp] = counts.get(sp, 0.0) + coeff
        return counts

    @staticmethod
    def _net_stoich(reactants: Dict[str, float], products: Dict[str, float]) -> Dict[str, float]:
        nu: Dict[str, float] = {}
        for s, c in products.items():
            if s != "M":
                nu[s] = nu.get(s, 0.0) + float(c)
        for s, c in reactants.items():
            if s != "M":
                nu[s] = nu.get(s, 0.0) - float(c)
        return nu

    def _load_mechanism(self) -> None:
        base = SurfaceChemistryLoader(self.yaml_file).load()
        species = set(base["species"])
        atomic_masses = dict(base.get("atomic_masses") or {})
        entries = list(base["reaction_entries"])

        self.species_all = sorted(species)
        self.atomic_masses = atomic_masses

        if len(entries) < 1:
            raise ValueError("No reactions found.")
        if len(entries) % 2 != 0:
            entries = entries[:-1]

        self.reactions: List[ReactionPair] = []
        for i in range(0, len(entries), 2):
            fwd = entries[i]
            rev = entries[i + 1]
            eq_f = str(fwd.get("equation", "N/A")).strip()
            eq_r = str(rev.get("equation", "N/A")).strip()
            f_valid = (eq_f.upper() != "N/A") and ("=>" in eq_f)
            if not f_valid:
                continue

            lhs, rhs = eq_f.split("=>", 1)
            reactants, m_order_f = SurfaceChemistryLoader._parse_equation_side(lhs)
            products, _m_rhs = SurfaceChemistryLoader._parse_equation_side(rhs)

            r_valid = (eq_r.upper() != "N/A") and ("=>" in eq_r)
            m_order_r = 0.0
            if r_valid:
                try:
                    lhs_r, _ = eq_r.split("=>", 1)
                    _rc, m_order_r = SurfaceChemistryLoader._parse_equation_side(lhs_r)
                except Exception:
                    m_order_r = 0.0

            kf = SurfaceChemistryLoader._parse_rate_constant_dict(fwd.get("rate-constant", {}))
            kr = SurfaceChemistryLoader._parse_rate_constant_dict(rev.get("rate-constant", {})) if r_valid else (0.0, 0.0, 0.0)

            order_f = sum(reactants.values()) + float(m_order_f)
            order_r = sum(products.values()) + float(m_order_r)
            kf = self._arrhenius_to_si(kf, order_f)
            kr = self._arrhenius_to_si(kr, order_r) if r_valid else (0.0, 0.0, 0.0)

            imp = None
            imp = fwd.get("impingements", None)
            if imp is None and isinstance(fwd, dict) and "impingement" in fwd:
                imp = fwd.get("impingement")

            eff_f = dict(fwd.get("efficiencies") or {})
            eff_r = dict(rev.get("efficiencies") or {}) if r_valid else {}

            rp = ReactionPair(
                equation_fwd=eq_f,
                reactants=reactants,
                products=products,
                m_order_fwd=float(m_order_f),
                m_order_rev=float(m_order_r),
                eff_fwd={str(k): float(v) for k, v in eff_f.items() if isinstance(v, (int, float))},
                eff_rev={str(k): float(v) for k, v in eff_r.items() if isinstance(v, (int, float))},
                kf=kf,
                kr=kr,
                has_reverse=r_valid,
                impingement=imp if imp is not None else None,
            )
            self.reactions.append(rp)

        if not self.reactions:
            raise ValueError(f"No valid reactions loaded from {self.yaml_file}")

    def _arrhenius_k(self, A: float, b: float, Ea: float, T: float) -> float:
        if A == 0.0:
            return 0.0
        return float(A) * (float(T) ** float(b)) * math.exp(-float(Ea) / (self.R * float(T)))

    def _mixture_mw_kg_per_mol(self, y_gas: np.ndarray, background_phi: Optional[Dict[str, float]] = None) -> float:
        y_full: Dict[str, float] = {}
        if background_phi:
            for sp, v in background_phi.items():
                try:
                    y_full[str(sp)] = float(v)
                except Exception:
                    continue
        for sp, idx in self.gas_index.items():
            y_full[sp] = float(y_gas[idx])

        def is_gas(sp: str) -> bool:
            if sp in ("S", "M"):
                return False
            if self._is_surface_species(sp):
                return False
            if sp in self.constant_species:
                return False
            return True

        denom = 0.0
        for sp, v in y_full.items():
            if not is_gas(sp):
                continue
            if v > 0.0:
                denom += v
        if denom <= 0.0:
            return 0.0280134

        mw = 0.0
        for sp, v in y_full.items():
            if not is_gas(sp) or v <= 0.0:
                continue
            if sp not in self._mw_by_species:
                try:
                    self._mw_by_species[sp] = molecular_weight(sp, self.atomic_masses)
                except Exception:
                    continue
            mw += (v / denom) * (self._mw_by_species[sp] / 1000.0)
        return mw if mw > 0.0 else 0.0280134

    def _species_conc_for_rate(
        self,
        sp: str,
        *,
        y_gas: np.ndarray,
        theta: np.ndarray,
        C_tot: float,
        theta_S: float,
    ) -> float:
        if sp in self.constant_species:
            return float(self.constant_species[sp])
        if sp == "S":
            return float(theta_S)
        if self._is_surface_species(sp):
            return float(theta[self.surface_index[sp]]) if sp in self.surface_index else 0.0
        if sp in self.gas_index:
            return float(y_gas[self.gas_index[sp]] * C_tot)
        return 0.0

    def _impingement_k(
        self,
        T: float,
        P: float,
        C_tot: float,
        y_gas: np.ndarray,
        imp: Dict[str, Any],
        *,
        background_phi: Optional[Dict[str, float]] = None,
        mw_kg_per_mol: Optional[float] = None,
    ) -> float:
        sticking = float(imp.get("sticking", 1.0)) if isinstance(imp, dict) else 1.0
        A_star = float(imp.get("A_star", self.A_star)) if isinstance(imp, dict) else self.A_star
        if A_star < 0.0:
            A_star = 0.0
        if A_star > 1.0:
            A_star = 1.0
        if mw_kg_per_mol is None or (not np.isfinite(mw_kg_per_mol)) or mw_kg_per_mol <= 0.0:
            mw_kg_per_mol = self._mixture_mw_kg_per_mol(y_gas, background_phi=background_phi)
        m_imp = float(mw_kg_per_mol) / self.NA
        root = math.sqrt(max(1e-300, 2.0 * math.pi * m_imp * self.kB * T))
        if self.impingement_model == "paper":
            n_tot = max(1e-300, float(self.NA) * float(max(C_tot, 1e-300)))
            denom = self.rho_s0 * root * n_tot
            return sticking * A_star * (P / denom)
        denom = self.rho_s0 * root * max(C_tot, 1e-300)
        return sticking * (P / denom)

    def _effective_M_conc(self, y_gas: np.ndarray, C_tot: float, eff: Dict[str, float]) -> float:
        if not eff:
            return C_tot
        M_eff = 0.0
        for sp, eps in eff.items():
            if sp in self.gas_index:
                M_eff += float(eps) * float(y_gas[self.gas_index[sp]] * C_tot)
        return M_eff if M_eff > 0.0 else C_tot

    def _reaction_rate_site(
        self,
        rxn: ReactionPair,
        *,
        T: float,
        P: float,
        C_tot: float,
        y_gas: np.ndarray,
        theta: np.ndarray,
        theta_S: float,
        background_phi: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float, float]:
        A_f, b_f, Ea_f = rxn.kf
        A_r, b_r, Ea_r = rxn.kr
        if rxn.impingement is not None:
            mw_imp_kg_per_mol: Optional[float] = None
            for sp in rxn.reactants.keys():
                if sp in ("S", "M") or self._is_surface_species(sp) or (sp in self.constant_species):
                    continue
                if sp in self._mw_by_species:
                    mw_imp_kg_per_mol = float(self._mw_by_species[sp]) / 1000.0
                else:
                    try:
                        mw_imp_kg_per_mol = molecular_weight(sp, self.atomic_masses) / 1000.0
                    except Exception:
                        mw_imp_kg_per_mol = None
                break
            kf = self._impingement_k(
                T,
                P,
                C_tot,
                y_gas,
                rxn.impingement,
                background_phi=background_phi,
                mw_kg_per_mol=mw_imp_kg_per_mol,
            )
        else:
            kf = self._arrhenius_k(A_f, b_f, Ea_f, T)
        kr = self._arrhenius_k(A_r, b_r, Ea_r, T) if rxn.has_reverse else 0.0

        rf = kf
        for sp, nu in rxn.reactants.items():
            c = self._species_conc_for_rate(sp, y_gas=y_gas, theta=theta, C_tot=C_tot, theta_S=theta_S)
            if (rxn.impingement is not None) and (self.impingement_model == "paper") and (sp not in ("S", "M")) and (not self._is_surface_species(sp)):
                c *= self.NA
            if c <= 0.0:
                return 0.0, 0.0, 0.0
            rf *= c ** float(nu)

        rr = kr
        if rxn.has_reverse and kr > 0.0:
            for sp, nu in rxn.products.items():
                c = self._species_conc_for_rate(sp, y_gas=y_gas, theta=theta, C_tot=C_tot, theta_S=theta_S)
                if c <= 0.0:
                    rr = 0.0
                    break
                rr *= c ** float(nu)
        else:
            rr = 0.0

        if rxn.m_order_fwd > 0.0:
            M_eff = self._effective_M_conc(y_gas, C_tot, rxn.eff_fwd)
            rf *= M_eff ** rxn.m_order_fwd
        if rxn.has_reverse and rxn.m_order_rev > 0.0:
            M_eff = self._effective_M_conc(y_gas, C_tot, rxn.eff_rev)
            rr *= M_eff ** rxn.m_order_rev
        return rf, rr, (rf - rr)

    def ode_system(self, t: float, y: np.ndarray, *, T: float, P: float, fixed_phi: Dict[str, float]) -> np.ndarray:
        n_gas = len(self.gas_species)
        y_gas = np.clip(y[:n_gas], 0.0, 1.0)
        theta = np.clip(y[n_gas:], 0.0, 1.0)
        theta_sum = float(np.sum(theta))
        theta_S = max(0.0, 1.0 - theta_sum)
        C_tot = P / (self.R * T)
        scale = self.a_cat * (self.rho_s0 / self.NA) / max(C_tot, 1e-300)
        dy_gas = np.zeros_like(y_gas)
        dtheta = np.zeros_like(theta)
        for k, rxn in enumerate(self.reactions):
            _rf, _rr, rnet = self._reaction_rate_site(
                rxn,
                T=T,
                P=P,
                C_tot=C_tot,
                y_gas=y_gas,
                theta=theta,
                theta_S=theta_S,
                background_phi=fixed_phi,
            )
            if rnet == 0.0:
                continue
            nu_s = self._rxn_nu_surf[k]
            for sp, nu in nu_s.items():
                dtheta[self.surface_index[sp]] += float(nu) * rnet
            nu_g = self._rxn_nu_gas[k]
            for sp, nu in nu_g.items():
                dy_gas[self.gas_index[sp]] += scale * float(nu) * rnet
        return np.concatenate([dy_gas, dtheta])

    def compute_carbon_fluxes(self, sol, *, T_final: float, P_atm: float) -> Tuple[float, float]:
        T = float(T_final)
        P = float(P_atm) * 101325.0
        y_end = np.asarray(sol.y)[:, -1].astype(float, copy=False)
        n_gas = len(self.gas_species)
        y_gas = np.clip(y_end[:n_gas], 0.0, 1.0)
        theta = np.clip(y_end[n_gas:], 0.0, 1.0)
        theta_S = max(0.0, 1.0 - float(np.sum(theta)))
        C_tot = P / (self.R * T)

        r_C2_s = 0.0
        r_C4_s = 0.0
        for rxn in self.reactions:
            if "C_NT" not in rxn.net_stoich:
                continue
            _rf, _rr, rnet = self._reaction_rate_site(
                rxn,
                T=T,
                P=P,
                C_tot=C_tot,
                y_gas=y_gas,
                theta=theta,
                theta_S=theta_S,
            )
            flux = (self.rho_s0 / self.NA) * float(rnet)
            if "C2_s" in rxn.reactants:
                r_C2_s += flux
            elif "C4_s" in rxn.reactants:
                r_C4_s += flux
        return float(r_C2_s), float(r_C4_s)

    def compute_cnt_growth_rate(
        self,
        sol,
        *,
        T_final: float,
        P_atm: float,
        d_outer_m: float,
        rho_part_m2: float,
        rho_c: float = 2200.0,
    ) -> Tuple[float, float, float]:
        r_C2_s, r_C4_s = self.compute_carbon_fluxes(sol, T_final=T_final, P_atm=P_atm)
        M_C = 0.012
        V_C = M_C / (float(rho_c) * self.NA)
        d_inner_m = 0.5 * float(d_outer_m)
        carbon_atom_flux_mol_m2_s = 2.0 * float(r_C2_s) + 4.0 * float(r_C4_s)
        V_deposited = self.NA * carbon_atom_flux_mol_m2_s * V_C / float(rho_part_m2)
        area_diff = (math.pi / 4.0) * (float(d_outer_m) ** 2 - float(d_inner_m) ** 2)
        u_cnt_m_per_s = V_deposited / max(area_diff, 1e-300)
        u_cnt_microns_per_s = u_cnt_m_per_s * 1e6
        ln_growth = math.log(max(u_cnt_microns_per_s, 1e-300))
        return float(u_cnt_m_per_s), float(u_cnt_microns_per_s), float(ln_growth)
# Constants
NAVA = 6.022e23  # Avogadro's number
kb = 1.38065e-23  # Boltzmann constant, J/K

# Catalyst parameters
# Puretzky dashed curve uses dp = 10 nm (paper reference)
dp = 10e-9  # catalyst particle diameter in meters
# Ma dotted curve uses dp = 10 nm 
dp_ma_line = 10e-9
n_m = 1e19  # surface density of carbon monolayer, atoms/m^2
alpha = 1  # number of monolayers

# Ma-method dp sweep (for shaded band)
# User-specified: dp in [5, 15] nm with interval 3 nm.
dp_sweep_nm = np.arange(5.0, 15.0, 3.0)  # -> 5, 8, 11, 14
if dp_sweep_nm[-1] < 15.0:
    dp_sweep_nm = np.append(dp_sweep_nm, 15.0)  # ensure max bound included
dp_sweep_m = dp_sweep_nm * 1e-9

# Species properties
species_weight = 4.33e-26  # mass of C2H2 in kg
species_weight_C2H4 = 8.65e-26 
P_FEEDSTOCK = 266.645  # partial pressure in Pa (2 Torr)

# Activation energies (eV)
E_a1 = 0.41  # activation energy for surface carbon formation

def calculate_k_sb(temperature_kelvin):
    E_sb = 2.2      # Activation energy in eV
    # Known value: k_sb = 17 s^-1 at T = 575°C (848.15 K)
    k_sb_known = 17
    T_known = 575 + 273.15  # Convert to Kelvin
    # Calculate pre-exponential factor B
    B = k_sb_known / np.exp(-E_sb / (8.617e-5 * T_known))
    # Calculate k_sb using Arrhenius equation
    k_sb = B * np.exp(-E_sb / (8.617e-5 * temperature_kelvin))
    return k_sb 

# Define the ODE system for Puretzky method
def ode_system_puretzky(t, y, temp_C, dp_m: float):
    N_c, N_L1, N_L2, N_b, N_t = y
    
    # Temperature in Kelvin
    temp = temp_C + 273.15
    dp_m = float(dp_m)
    S_0_local = np.pi * (dp_m) ** 2  # catalyst surface area in m^2
    
    # Calculate fluxes
    g_t = 1 - np.exp(-0.02 * t)
    
    n0 = NAVA * P_FEEDSTOCK / (8.314 * temp)
    P_pref = 1e-14
    E_p = 2.60
    n_p = (P_pref*np.exp((- E_p) / (8.617e-5 * temp)))*(g_t**2)*(n0**2)
    
    #have not considered the catalytic impact 
    n = (g_t*n0)-(2*n_p)
    
    adjustment_factor = 1e0
    A_etching = adjustment_factor*(6.91/6)*1e-13*np.exp(-72100/8.314/temp)
    B_etching = 3.5e-12*np.exp(-118100/8.314/temp)
    if n <= 0:
        n = 0
    
    F_b1 = 0.25 * S_0_local * (n) * ((kb * temp / (2 * np.pi * species_weight)) ** 0.5)
    F_c1 = F_b1 * 1*np.exp(-E_a1 / (8.617e-5 * temp))

    F_b2 = 0.25 * S_0_local * (n_p) * ((kb * temp / (2 * np.pi * species_weight_C2H4)) ** 0.5)
    F_c2 = F_b2 * 1*np.exp(-E_a1 / (8.617e-5 * temp))
    
    E_b = 1.5  # activation energy for bulk diffusion
    # Rate constants
    k_c1 = 3e-3  # carbonaceous layer formation, 1/s
    k_t = (4 * 1e-5 / (dp_m**2)) * np.exp((- E_b) / (8.617e-5 * temp))  # carbon diffusion to nanotube, 1/s
    k_d2 = 0  # catalyst reactivation rate (negligible in this model), 1/s
    k_d1 = 1

    # Gas front factor (models the leading edge of acetylene gas flow)
    E_rl = 2.4
    # Total layer coverage
    N_L = N_L1 + N_L2
    k_r = 1.2e8*np.exp((- E_rl) / (8.617e-5 * temp))
    etching_factor = (1e0*(A_etching*((0.966 * 101325 )**2)/(1+(B_etching*(0.966 * 101325 )))))

    dN_c_dt = F_c1 * (1 - N_L/(alpha * S_0_local * n_m)) - (calculate_k_sb(temp) + k_c1) * N_c-  (etching_factor* (N_c))
    dN_L1_dt = F_c2 * (1 - N_L/(alpha * S_0_local * n_m)) + k_c1 * N_c - k_d1*N_L2   -  (etching_factor* (N_L1))
    dN_L2_dt = (k_r * (1 - N_L / (alpha * S_0_local * n_m)) -k_d2 * N_L2)#+(etching_factor* (N_t))
    dN_b_dt = calculate_k_sb(temp) * N_c - k_t * N_b + k_d1*N_L1
    dN_t_dt = k_t * N_b
    
    return [dN_c_dt, dN_L1_dt, dN_L2_dt, dN_b_dt, dN_t_dt]

# Function to convert number of carbon atoms to CNT length in micrometers
def calculate_length_puretzky(N_t, dp_m: float):
    # Paper reference conversion (kept constant):
    # 7e6 carbon atoms per micrometer for a 10 nm diameter CNT (from the paper).
    _atoms_per_micrometer = 7e6
    return N_t / _atoms_per_micrometer


def compute_zone_times_from_geometry(
    *,
    T_target_K: float,
    T_inlet_K: float,
    L_ramp_m: float,
    L_uniform_m: float,
    u_inlet_m_s: float,
) -> tuple[float, float]:
    Tt = float(T_target_K)
    Tin = float(T_inlet_K)
    if Tin <= 0.0 or Tt <= 0.0:
        raise ValueError("Temperatures must be in Kelvin and > 0")
    if L_ramp_m < 0.0 or L_uniform_m < 0.0:
        raise ValueError("Lengths must be >= 0")
    u0 = float(u_inlet_m_s)
    if u0 <= 0.0:
        raise ValueError("u_inlet_m_s must be > 0")

    u_target = u0 * (Tt / Tin)
    t_uniform = (float(L_uniform_m) / u_target) if L_uniform_m > 0.0 else 0.0

    if L_ramp_m <= 0.0:
        t_ramp = 0.0
    elif abs(Tt - Tin) < 1e-15:
        t_ramp = float(L_ramp_m) / u0
    else:
        t_ramp = (float(L_ramp_m) * Tin) / (u0 * (Tt - Tin)) * math.log(Tt / Tin)

    return float(t_ramp), float(t_uniform)


def run_cantera_zone1_temperature_ramp(
    *,
    here: Path,
    T_inlet_K: float,
    T_target_K: float,
    t_end_s: float,
    P_atm: float,
    mechanism_yaml: str,
    initial_mole_fractions: Dict[str, float],
    n_steps: int = 400,
) -> Dict[str, float]:
    import cantera as ct

    def clamp_comp(comp: Dict[str, float]) -> Dict[str, float]:
        out = {k: (float(v) if float(v) > 0.0 else 0.0) for k, v in comp.items()}
        s = float(sum(out.values()))
        if s > 0.0 and abs(s - 1.0) > 1e-15:
            out = {k: v / s for k, v in out.items()}
        return out

    t_end = float(t_end_s)
    if t_end < 0.0:
        raise ValueError("t_end_s must be >= 0")
    if t_end == 0.0:
        return clamp_comp(dict(initial_mole_fractions))

    mech_path = (here / mechanism_yaml).resolve()
    gas = ct.Solution(str(mech_path))
    P_pa = float(P_atm) * float(ct.one_atm)
    gas.TPX = float(T_inlet_K), P_pa, clamp_comp(dict(initial_mole_fractions))

    r = ct.IdealGasReactor(gas, energy="off")
    net = ct.ReactorNet([r])

    N = int(max(1, n_steps))
    for k in range(N):
        t_next = (k + 1) * (t_end / N)
        T_next = float(T_inlet_K) + (float(T_target_K) - float(T_inlet_K)) * (t_next / t_end)
        r.thermo.TP = float(T_next), P_pa
        r.syncState()
        net.reinitialize()
        net.advance(float(t_next))

    return {sp: float(x) for sp, x in zip(gas.species_names, gas.X)}


def calculate_gamma(temp_K: float) -> float:
    temp_C = float(temp_K) - 273.15
    if temp_C <= 700.0:
        gamma = 6.382123e-03 * math.exp(1.113002e-02 * (float(temp_K) - 823.81))
    else:
        gamma = 3.461503e-02 * math.exp(-3.268908e-02 * (float(temp_K) - 973.00))
    return float(gamma)


def compute_wall_area_and_volume_cylinder(*, D_m: float, L_m: float) -> tuple[float, float]:
    if D_m <= 0 or L_m <= 0:
        raise ValueError("D_m and L_m must be > 0")
    A_wall = math.pi * D_m * L_m
    V = math.pi * (D_m * 0.5) ** 2 * L_m
    return A_wall, V


def sites_per_particle_from_reference(d_m: float, N_sites_ref: int = 250, d_ref_m: float = 5e-9) -> float:
    return N_sites_ref * (d_m / d_ref_m) ** 2


def compute_rho_s0_wall_based(
    *, rho_part_m2: float, d_np_m: float, N_sites_ref: int = 250, d_ref_m: float = 5e-9
) -> float:
    return sites_per_particle_from_reference(d_np_m, N_sites_ref, d_ref_m) * rho_part_m2


def build_stoichiometric_arrays(sim: ParametricSurfaceReactorSimulation, T_final: float):
    species_list = sim.gas_species + sim.surface_species + ["S"]
    species_index = {sp: i for i, sp in enumerate(species_list)}
    n_species = len(species_list)
    n_gas = len(sim.gas_species)
    n_surf = len(sim.surface_species)
    reactions = sim.reactions
    n_rxn = len(reactions)

    max_react = max(len(r.reactants) for r in reactions)
    max_prod = max(len(r.products) for r in reactions)
    max_eff = max(len(r.eff_fwd) for r in reactions) if reactions else 0
    max_eff = max(max_eff, max(len(r.eff_rev) for r in reactions) if reactions else 0)

    reactant_idx = np.full((n_rxn, max_react), -1, dtype=np.intc)
    reactant_nu = np.zeros((n_rxn, max_react), dtype=np.float64)
    product_idx = np.full((n_rxn, max_prod), -1, dtype=np.intc)
    product_nu = np.zeros((n_rxn, max_prod), dtype=np.float64)
    eff_f_idx = np.full((n_rxn, max_eff), -1, dtype=np.intc)
    eff_f_val = np.zeros((n_rxn, max_eff), dtype=np.float64)
    eff_r_idx = np.full((n_rxn, max_eff), -1, dtype=np.intc)
    eff_r_val = np.zeros((n_rxn, max_eff), dtype=np.float64)
    eff_counts_f = np.zeros(n_rxn, dtype=np.intc)
    eff_counts_r = np.zeros(n_rxn, dtype=np.intc)

    net_stoich = np.zeros((n_rxn, n_species), dtype=np.float64)
    A_f_arr = np.zeros(n_rxn, dtype=np.float64)
    b_f_arr = np.zeros(n_rxn, dtype=np.float64)
    Ea_f_arr = np.zeros(n_rxn, dtype=np.float64)
    A_r_arr = np.zeros(n_rxn, dtype=np.float64)
    b_r_arr = np.zeros(n_rxn, dtype=np.float64)
    Ea_r_arr = np.zeros(n_rxn, dtype=np.float64)
    m_order_fwd = np.zeros(n_rxn, dtype=np.float64)
    m_order_rev = np.zeros(n_rxn, dtype=np.float64)

    imp_flag = np.zeros(n_rxn, dtype=np.intc)
    imp_sticking = np.ones(n_rxn, dtype=np.float64)
    imp_A_star = np.ones(n_rxn, dtype=np.float64) * sim.A_star
    imp_mw_kg_per_mol = np.zeros(n_rxn, dtype=np.float64)

    for i, rxn in enumerate(reactions):
        slot = 0
        for sp, coeff in rxn.reactants.items():
            if sp == "M":
                continue
            idx = species_index.get(sp)
            if idx is None:
                continue
            reactant_idx[i, slot] = idx
            reactant_nu[i, slot] = coeff
            slot += 1

        slot = 0
        for sp, coeff in rxn.products.items():
            if sp == "M":
                continue
            idx = species_index.get(sp)
            if idx is None:
                continue
            product_idx[i, slot] = idx
            product_nu[i, slot] = coeff
            slot += 1

        m_order_fwd[i] = rxn.m_order_fwd
        m_order_rev[i] = rxn.m_order_rev
        A_f_arr[i], b_f_arr[i], Ea_f_arr[i] = rxn.kf
        A_r_arr[i], b_r_arr[i], Ea_r_arr[i] = rxn.kr

        eff_slot = 0
        for sp, eps in rxn.eff_fwd.items():
            idx = species_index.get(sp)
            if idx is None:
                continue
            eff_f_idx[i, eff_slot] = idx
            eff_f_val[i, eff_slot] = eps
            eff_slot += 1
        eff_counts_f[i] = eff_slot

        eff_slot = 0
        for sp, eps in rxn.eff_rev.items():
            idx = species_index.get(sp)
            if idx is None:
                continue
            eff_r_idx[i, eff_slot] = idx
            eff_r_val[i, eff_slot] = eps
            eff_slot += 1
        eff_counts_r[i] = eff_slot

        imp = rxn.impingement
        if imp is not None:
            imp_flag[i] = 1
            imp_sticking[i] = float(imp.get("sticking", 1.0))
            imp_A_star[i] = float(imp.get("A_star", sim.A_star))
            mw_here = 0.0
            for sp in rxn.reactants.keys():
                if sp in ("S", "M") or sim._is_surface_species(sp) or (sp in sim.constant_species):
                    continue
                try:
                    if sp in sim._mw_by_species:
                        mw_here = float(sim._mw_by_species[sp]) / 1000.0
                    else:
                        mw_here = float(molecular_weight(sp, sim.atomic_masses)) / 1000.0
                except Exception:
                    mw_here = 0.0
                break
            imp_mw_kg_per_mol[i] = mw_here

        for sp, nu in rxn.net_stoich.items():
            if sp == "M":
                continue
            idx = species_index.get(sp)
            if idx is None:
                continue
            net_stoich[i, idx] = nu

    return {
        "species_index": species_index,
        "n_species": n_species,
        "n_gas": n_gas,
        "n_surf": n_surf,
        "net_stoich": net_stoich,
        "reactant_idx": reactant_idx,
        "reactant_nu": reactant_nu,
        "product_idx": product_idx,
        "product_nu": product_nu,
        "A_f": A_f_arr,
        "b_f": b_f_arr,
        "Ea_f": Ea_f_arr,
        "A_r": A_r_arr,
        "b_r": b_r_arr,
        "Ea_r": Ea_r_arr,
        "m_order_fwd": m_order_fwd,
        "m_order_rev": m_order_rev,
        "eff_f_idx": eff_f_idx,
        "eff_f_val": eff_f_val,
        "eff_r_idx": eff_r_idx,
        "eff_r_val": eff_r_val,
        "eff_counts_f": eff_counts_f,
        "eff_counts_r": eff_counts_r,
        "imp_flag": imp_flag,
        "imp_sticking": imp_sticking,
        "imp_A_star": imp_A_star,
        "imp_mw_kg_per_mol": imp_mw_kg_per_mol,
        "species_list": species_list,
    }


def _arrhenius_kernel(A: float, b: float, Ea: float, T: float) -> float:
    if A == 0.0:
        return 0.0
    return float(A) * (float(T) ** float(b)) * math.exp(-float(Ea) / (8.314462618 * float(T)))


def compute_dydt(
    conc,
    T,
    P,
    C_tot,
    scale,
    n_gas,
    n_surf,
    net_stoich,
    reactant_idx,
    reactant_nu,
    product_idx,
    product_nu,
    A_f,
    b_f,
    Ea_f,
    A_r,
    b_r,
    Ea_r,
    m_order_f,
    m_order_r,
    eff_f_idx,
    eff_f_val,
    eff_r_idx,
    eff_r_val,
    eff_counts_f,
    eff_counts_r,
    imp_flag,
    imp_sticking,
    imp_A_star,
    imp_mw_kg_per_mol,
    rho_s0,
    imp_model_paper,
    mw_mix_kg_per_mol,
    freeze_gas,
):
    conc = np.asarray(conc, dtype=float)
    net_stoich = np.asarray(net_stoich, dtype=float)
    reactant_idx = np.asarray(reactant_idx, dtype=int)
    reactant_nu = np.asarray(reactant_nu, dtype=float)
    product_idx = np.asarray(product_idx, dtype=int)
    product_nu = np.asarray(product_nu, dtype=float)
    A_f = np.asarray(A_f, dtype=float)
    b_f = np.asarray(b_f, dtype=float)
    Ea_f = np.asarray(Ea_f, dtype=float)
    A_r = np.asarray(A_r, dtype=float)
    b_r = np.asarray(b_r, dtype=float)
    Ea_r = np.asarray(Ea_r, dtype=float)
    m_order_f = np.asarray(m_order_f, dtype=float)
    m_order_r = np.asarray(m_order_r, dtype=float)
    eff_f_idx = np.asarray(eff_f_idx, dtype=int)
    eff_f_val = np.asarray(eff_f_val, dtype=float)
    eff_r_idx = np.asarray(eff_r_idx, dtype=int)
    eff_r_val = np.asarray(eff_r_val, dtype=float)
    eff_counts_f = np.asarray(eff_counts_f, dtype=int)
    eff_counts_r = np.asarray(eff_counts_r, dtype=int)
    imp_flag = np.asarray(imp_flag, dtype=int)
    imp_sticking = np.asarray(imp_sticking, dtype=float)
    imp_A_star = np.asarray(imp_A_star, dtype=float)
    imp_mw_kg_per_mol = np.asarray(imp_mw_kg_per_mol, dtype=float)

    n_rxn = A_f.shape[0]
    max_react = reactant_idx.shape[1]
    max_prod = product_idx.shape[1]
    dy_gas = np.zeros(int(n_gas), dtype=float)
    dtheta = np.zeros(int(n_surf), dtype=float)

    if mw_mix_kg_per_mol <= 0.0:
        mw_mix_kg_per_mol = 0.0280134
    NA_const = 6.02214076e23
    kB_const = 1.380649e-23
    n_tot = max(1e-300, NA_const * float(C_tot))

    for i in range(n_rxn):
        if imp_flag[i] != 0:
            mw_use = float(mw_mix_kg_per_mol)
            if imp_mw_kg_per_mol is not None and imp_mw_kg_per_mol.shape[0] == n_rxn:
                if imp_mw_kg_per_mol[i] > 0.0:
                    mw_use = float(imp_mw_kg_per_mol[i])
            m_use = mw_use / NA_const
            root = math.sqrt(max(1e-300, 2.0 * math.pi * m_use * kB_const * float(T)))
            if imp_model_paper:
                kf_i = float(imp_sticking[i]) * float(imp_A_star[i]) * (float(P) / (float(rho_s0) * root * n_tot))
            else:
                kf_i = float(imp_sticking[i]) * (float(P) / (float(rho_s0) * root * max(float(C_tot), 1e-300)))
        else:
            kf_i = _arrhenius_kernel(A_f[i], b_f[i], Ea_f[i], T)

        prod_f = 1.0
        for slot in range(max_react):
            idx = int(reactant_idx[i, slot])
            if idx < 0:
                break
            if (imp_flag[i] != 0) and imp_model_paper and idx < int(n_gas):
                prod_f *= (conc[idx] * NA_const) ** float(reactant_nu[i, slot])
            else:
                prod_f *= conc[idx] ** float(reactant_nu[i, slot])
        rf = kf_i * prod_f

        if m_order_f[i] > 0.0:
            if eff_counts_f[i] <= 0:
                M_eff = float(C_tot)
            else:
                M_eff = 0.0
                for slot in range(int(eff_counts_f[i])):
                    idx = int(eff_f_idx[i, slot])
                    if idx < 0 or idx >= int(n_gas):
                        continue
                    M_eff += float(eff_f_val[i, slot]) * float(conc[idx])
                if M_eff <= 0.0:
                    M_eff = float(C_tot)
            rf *= M_eff ** float(m_order_f[i])

        rr = 0.0
        if A_r[i] != 0.0:
            kr_i = _arrhenius_kernel(A_r[i], b_r[i], Ea_r[i], T)
            if kr_i != 0.0:
                prod_r = 1.0
                for slot in range(max_prod):
                    idx = int(product_idx[i, slot])
                    if idx < 0:
                        break
                    prod_r *= conc[idx] ** float(product_nu[i, slot])
                rr = kr_i * prod_r

                if m_order_r[i] > 0.0:
                    if eff_counts_r[i] <= 0:
                        M_eff = float(C_tot)
                    else:
                        M_eff = 0.0
                        for slot in range(int(eff_counts_r[i])):
                            idx = int(eff_r_idx[i, slot])
                            if idx < 0 or idx >= int(n_gas):
                                continue
                            M_eff += float(eff_r_val[i, slot]) * float(conc[idx])
                        if M_eff <= 0.0:
                            M_eff = float(C_tot)
                    rr *= M_eff ** float(m_order_r[i])

        rnet = rf - rr
        if rnet == 0.0:
            continue

        if not freeze_gas:
            for j in range(int(n_gas)):
                if net_stoich[i, j] != 0.0:
                    dy_gas[j] += float(scale) * float(net_stoich[i, j]) * rnet
        for j in range(int(n_surf)):
            if net_stoich[i, int(n_gas) + j] != 0.0:
                dtheta[j] += float(net_stoich[i, int(n_gas) + j]) * rnet

    return np.concatenate([dy_gas, dtheta])


if _compute_dydt_fast is not None:
    compute_dydt = _compute_dydt_fast


# Ma et al. method for growth rate calculation
class MaGrowthModel:
    """
    Surface-kinetics growth model.

    This class integrates the logic from `cython_driver_trusted_rate.py` into this plotting script,
    using the same API expected by the rest of the file:
      - calculate_growth_over_time(temp_K, time_span) -> (time_points, lengths_um)
      - calculate_terminal_length(temp_K, time_duration=...) -> final length (um)

    Important behavior (matches your latest driver requirements):
    - Zone 1: gas-only Cantera with a 25°C -> T_target linear temperature ramp over L_ramp.
    - Zone 2: uniform at T_target over L_uniform, gas composition is frozen to the Zone-1 outlet
      (no Cantera advance, no surface-to-gas updates).
    """

    def __init__(
        self,
        *,
        dp: float,
        P_FEEDSTOCK: float,
        P_atm: float = 0.966,
        x_h2: float = 0.16,
        inert: str = "Ar",
        T_inlet_K: float = 298.15,
        L_ramp_m: float = 0.3429,
        L_uniform_m: float = 0.2032,
        u_inlet_25C_m_s: float = 0.005,
        dt_surf_s: float = 1.0,
        rho_part_m2: float = 1.0e15,
        surface_yaml: str = "CX_UI_Surface_Kinetics.yaml",
        gas_mech_yaml: str = "FFCM2.yaml",
        cantera_ramp_steps: int = 400,
        verbose: bool = False,
    ):
        self.dp_m = float(dp)
        self.P_FEEDSTOCK_Pa = float(P_FEEDSTOCK)
        self.P_atm = float(P_atm)
        self.x_h2 = float(x_h2)
        self.inert = str(inert)
        self.T_inlet_K = float(T_inlet_K)
        self.L_ramp_m = float(L_ramp_m)
        self.L_uniform_m = float(L_uniform_m)
        self.u_inlet_25C_m_s = float(u_inlet_25C_m_s)
        self.dt_surf_s = float(dt_surf_s)
        self.rho_part_m2 = float(rho_part_m2)
        self.surface_yaml = str(surface_yaml)
        self.gas_mech_yaml = str(gas_mech_yaml)
        self.cantera_ramp_steps = int(cantera_ramp_steps)
        self.verbose = bool(verbose)

      

    def _inlet_mole_fractions(self) -> dict:
        x_c2h2 = 0.0025
        x_h2 = 0.16
        x_inert = 1.0 - x_c2h2 - x_h2
        return {"C2H2": x_c2h2, "H2": x_h2, self.inert: x_inert}

    def calculate_growth_over_time(self, temp_K, time_span):
        """
        Returns:
          time_points (s), lengths_um (µm) for Zone 2 (uniform zone only).

        Note: Zone-1 ramp is used only to compute the gas composition entering Zone 2.
        """
        import cantera as ct

        here = Path(__file__).resolve().parent

        T_final = float(temp_K)
       

        # Residence times from furnace geometry
        t_ramp_s, t_uniform_s = compute_zone_times_from_geometry(
            T_target_K=T_final,
            T_inlet_K=float(self.T_inlet_K),
            L_ramp_m=float(self.L_ramp_m),
            L_uniform_m=float(self.L_uniform_m),
            u_inlet_m_s=float(self.u_inlet_25C_m_s),
        )

     
 
        if time_span is not None and len(time_span) == 2:
            try:
                t_req = float(time_span[1]) - float(time_span[0])
            except Exception:
                t_req = float("nan")
            if np.isfinite(t_req) and t_req > 0.0:
                t_uniform_s = float(t_req)

        if self.verbose:
            print(
                f"[model] T={T_final:.2f} K | t_ramp={t_ramp_s:.6g}s, t_uniform={t_uniform_s:.6g}s | dt={self.dt_surf_s}"
            )

        # Zone 1: Cantera with linear temperature ramp (25°C -> T_final)
        aa12_zone_in = run_cantera_zone1_temperature_ramp(
            here=here,
            T_inlet_K=float(self.T_inlet_K),
            T_target_K=T_final,
            t_end_s=float(t_ramp_s),
            P_atm=float(self.P_atm),
            mechanism_yaml=str(self.gas_mech_yaml),
            initial_mole_fractions=dict(self._inlet_mole_fractions()),
            n_steps=int(self.cantera_ramp_steps),
        )

        # Zone 2: surface integration with frozen gas composition
        gamma_T = calculate_gamma(T_final)
        d_np = float(self.dp_m)
        rho_s0_wall = compute_rho_s0_wall_based(rho_part_m2=float(self.rho_part_m2), d_np_m=d_np)
        A_star = catalyst_area_fraction_A_star(rho_part_m2=float(self.rho_part_m2), d_np_outer_m=d_np)

        surface_yaml_path = here / str(self.surface_yaml)
        if not surface_yaml_path.exists():
            raise FileNotFoundError(f"Missing surface mechanism YAML: {surface_yaml_path}")

       
        D_m = 0.0762
        L_m = 0.4445
        A_wall_m2, V_m3 = compute_wall_area_and_volume_cylinder(D_m=float(D_m), L_m=float(L_m))
        a_wall_m2_m3 = A_wall_m2 / V_m3

        sim = ParametricSurfaceReactorSimulation(
            yaml_file=str(surface_yaml_path),
            rho_s0_sites_m2=float(rho_s0_wall),
            a_cat_m2_m3=float(a_wall_m2_m3),
            impingement_model="paper",
            A_star=float(A_star) * float(gamma_T),
            inert_species=["N2", "Ar", "He"],
            constant_species={"NT": 1.0},
        )

        stoich = build_stoichiometric_arrays(sim, T_final)
        # --- dp correction for YAML Arrhenius prefactors (A) ---
        # Your YAML was calibrated at dp = 15 nm, so for other dp values:
        #   A_new = A_default * (15 / d_new)^2
        # where d_new is in nm (equivalently (15e-9 / d_new_m)^2).
        _dp_ref_m = 15e-9
        _dp_m = float(d_np)
        if _dp_m > 0.0:
            _A_scale = (_dp_ref_m / _dp_m) ** 2
            stoich["A_f"] = np.asarray(stoich["A_f"], dtype=float) * float(_A_scale)
            stoich["A_r"] = np.asarray(stoich["A_r"], dtype=float) * float(_A_scale)
            if self.verbose:
                print(f"[model] scaling YAML A by factor={_A_scale:.6g} for dp={_dp_m*1e9:.3f} nm")
        n_gas = int(stoich["n_gas"])
        n_surf = int(stoich["n_surf"])
        n_species = int(stoich["n_species"])

        P_pa = float(self.P_atm) * 101325.0
        C_tot = P_pa / (sim.R * T_final)
        scale = sim.a_cat * (sim.rho_s0 / sim.NA) / max(C_tot, 1e-300)

        def clamp_comp(comp: Dict[str, float]) -> Dict[str, float]:
            out = {k: (float(v) if float(v) > 0.0 else 0.0) for k, v in comp.items()}
            s = float(sum(out.values()))
            if s > 0.0 and abs(s - 1.0) > 1e-15:
                out = {k: v / s for k, v in out.items()}
            return out

        import cantera as ct  

        X_full_const: Dict[str, float] = clamp_comp(dict(aa12_zone_in))
        gas_const = ct.Solution(str((here / str(self.gas_mech_yaml)).resolve()))
        gas_const.TPX = float(T_final), float(self.P_atm) * float(ct.one_atm), X_full_const

        y_tracked_const = np.zeros(n_gas, dtype=float)
        for sp, idx in sim.gas_index.items():
            try:
                y_tracked_const[idx] = float(gas_const[sp].X[0])
            except Exception:
                y_tracked_const[idx] = 0.0
        mw_mix_kg_per_mol_const = float(gas_const.mean_molecular_weight) / 1000.0

        theta = np.zeros(n_surf, dtype=float)
        L_cnt_m = 0.0
        t = 0.0

        time_points = [0.0]
        lengths_um = [0.0]

        class SimpleSol:
            def __init__(self, tt, yy):
                self.t = np.array([tt])
                self.y = yy.reshape(-1, 1)

        while t < float(t_uniform_s) - 1e-15:
            dt = min(float(self.dt_surf_s), float(t_uniform_s) - t)
            y_tracked = y_tracked_const
            mw_mix_kg_per_mol = mw_mix_kg_per_mol_const

            # --- Surface kinetics over dt (gas frozen) ---
            def ode_theta_local(_t, th_local):
                th_local = np.clip(np.asarray(th_local, dtype=float), 0.0, 1.0)
                s_th = float(np.sum(th_local))
                if s_th > 1.0:
                    th_local = th_local / max(s_th, 1e-300)
                theta_S = max(0.0, 1.0 - float(np.sum(th_local)))
                conc = np.zeros(n_species, dtype=float)
                conc[:n_gas] = y_tracked * C_tot
                conc[n_gas : n_gas + n_surf] = th_local
                conc[-1] = theta_S
                dydt_full = compute_dydt(
                    conc,
                    T_final,
                    P_pa,
                    C_tot,
                    scale,
                    n_gas,
                    n_surf,
                    stoich["net_stoich"],
                    stoich["reactant_idx"],
                    stoich["reactant_nu"],
                    stoich["product_idx"],
                    stoich["product_nu"],
                    stoich["A_f"],
                    stoich["b_f"],
                    stoich["Ea_f"],
                    stoich["A_r"],
                    stoich["b_r"],
                    stoich["Ea_r"],
                    stoich["m_order_fwd"],
                    stoich["m_order_rev"],
                    stoich["eff_f_idx"],
                    stoich["eff_f_val"],
                    stoich["eff_r_idx"],
                    stoich["eff_r_val"],
                    stoich["eff_counts_f"],
                    stoich["eff_counts_r"],
                    stoich["imp_flag"],
                    stoich["imp_sticking"],
                    stoich["imp_A_star"],
                    stoich["imp_mw_kg_per_mol"],
                    sim.rho_s0,
                    sim.impingement_model == "paper",
                    mw_mix_kg_per_mol,
                    True,
                )
                return np.array(dydt_full[n_gas : n_gas + n_surf], dtype=float, copy=False)

            sol_local = solve_ivp(
                ode_theta_local,
                [t, t + dt],
                theta,
                method="BDF",
                rtol=1e-3,
                atol=1e-8,
                max_step=float(dt),
            )
            if not sol_local.success:
                raise RuntimeError(f"solve_ivp failed in growth zone at T={T_final} K: {sol_local.message}")
            theta = np.array(sol_local.y[:, -1], dtype=float, copy=False)
            theta = np.clip(theta, 0.0, 1.0)
            s_theta = float(np.sum(theta))
            if s_theta > 1.0:
                theta = theta / max(s_theta, 1e-300)

            # Integrate length
            sol_step = SimpleSol(float(t + dt), np.concatenate([y_tracked, theta]))
            u_m_s, _u_um_s, _ln = sim.compute_cnt_growth_rate(
                sol_step,
                T_final=T_final,
                P_atm=float(self.P_atm),
                d_outer_m=float(d_np),
                rho_part_m2=float(self.rho_part_m2),
                rho_c=2200.0,
            )
            L_cnt_m += float(u_m_s) * float(dt)

            t += float(dt)
            time_points.append(float(t))
            lengths_um.append(float(L_cnt_m) * 1e6)

        return np.array(time_points, dtype=float), np.array(lengths_um, dtype=float)
    
    def calculate_terminal_length(self, temp_K, time_duration=5000):
        """Calculate final CNT length at end of time period"""
        # NOTE: `time_duration` is treated as an optional override for the uniform-zone duration.
        # If you want strictly geometry-based residence time, call with time_duration=None.
        if time_duration is None:
            time_points, lengths = self.calculate_growth_over_time(temp_K, None)
        else:
            time_points, lengths = self.calculate_growth_over_time(temp_K, [0, time_duration])
        return lengths[-1]

# Function to read Excel bounds data for terminal length
def read_excel_bounds(filename):
    """Read lower and upper bounds from Excel file"""
    try:
        # Read Excel file
        df = pd.read_excel(filename, header=None)
        
        # Extract lower bound data (columns 0 and 1, starting from row 2)
        lower_temps = []
        lower_lengths = []
        upper_temps = []
        upper_lengths = []
        
        for i in range(2, len(df)):  # Skip header rows
            # Lower bound: columns A (0) and B (1)
            if pd.notna(df.iloc[i, 0]) and pd.notna(df.iloc[i, 1]):
                lower_lengths.append(df.iloc[i, 0])
                lower_temps.append(df.iloc[i, 1])
            
            # Upper bound: columns D (3) and E (4)
            if pd.notna(df.iloc[i, 3]) and pd.notna(df.iloc[i, 4]):
                upper_lengths.append(df.iloc[i, 3])
                upper_temps.append(df.iloc[i, 4])
        
        return np.array(lower_temps), np.array(lower_lengths), np.array(upper_temps), np.array(upper_lengths)
    
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        # Return empty arrays if file can't be read
        return np.array([]), np.array([]), np.array([]), np.array([])

# Function to read experimental time-series data
def read_experimental_data(filename):
    """Read experimental time-series data from Excel file"""
    try:
        df = pd.read_excel(filename, header=None)
        
        experimental_data = {}
        
        # Temperature columns mapping: [time_col, length_col]
        temp_columns = {
            575: [0, 1],  # Columns A, B
            600: [3, 4],  # Columns D, E
            650: [6, 7],  # Columns G, H
            700: [9, 10]  # Columns J, K
        }
        
        for temp, [time_col, length_col] in temp_columns.items():
            times = []
            lengths = []
            
            for i in range(2, len(df)):  # Start from row 2 (skip headers)
                time_val = df.iloc[i, time_col]
                length_val = df.iloc[i, length_col]
                
                # Only add if both values exist and are numbers
                if pd.notna(time_val) and pd.notna(length_val):
                    # Handle string values that might have extra characters
                    if isinstance(time_val, str):
                        try:
                            time_val = float(time_val.replace(',', ''))
                        except:
                            continue
                    if isinstance(length_val, str):
                        try:
                            length_val = float(length_val.replace(',', ''))
                        except:
                            continue
                    
                    times.append(float(time_val))
                    lengths.append(float(length_val))
            
            experimental_data[temp] = {
                'times': np.array(times),
                'lengths': np.array(lengths)
            }
        
        return experimental_data
    
    except Exception as e:
        print(f"Error reading experimental data: {e}")
        return {}

# NEW FUNCTION: Read bounds data for time-series plots
def read_time_series_bounds(filename):
    """Read lower and upper bounds for time-series data from Excel file"""
    try:
        df = pd.read_excel(filename, header=None)
        
        bounds_data = {}
        
        # Column mapping for each temperature
        # Based on the Excel structure: 575C (cols 0-5), 600C (cols 6-11), 650C (cols 12-17), 700C (cols 18-23)
        temp_column_mapping = {
            575: {'lower_length': 0, 'lower_time': 1, 'upper_length': 3, 'upper_time': 4},
            600: {'lower_length': 6, 'lower_time': 7, 'upper_length': 9, 'upper_time': 10},
            650: {'lower_length': 12, 'lower_time': 13, 'upper_length': 15, 'upper_time': 16},
            700: {'lower_length': 18, 'lower_time': 19, 'upper_length': 21, 'upper_time': 22}
        }
        
        for temp, cols in temp_column_mapping.items():
            lower_times = []
            lower_lengths = []
            upper_times = []
            upper_lengths = []
            
            for i in range(3, len(df)):  # Start from row 3 (skip headers)
                # Lower bounds
                if (pd.notna(df.iloc[i, cols['lower_time']]) and 
                    pd.notna(df.iloc[i, cols['lower_length']])):
                    lower_times.append(df.iloc[i, cols['lower_time']])
                    lower_lengths.append(df.iloc[i, cols['lower_length']])
                
                # Upper bounds
                if (pd.notna(df.iloc[i, cols['upper_time']]) and 
                    pd.notna(df.iloc[i, cols['upper_length']])):
                    upper_times.append(df.iloc[i, cols['upper_time']])
                    upper_lengths.append(df.iloc[i, cols['upper_length']])
            
            bounds_data[temp] = {
                'lower_times': np.array(lower_times),
                'lower_lengths': np.array(lower_lengths),
                'upper_times': np.array(upper_times),
                'upper_lengths': np.array(upper_lengths)
            }
        
        return bounds_data
    
    except Exception as e:
        print(f"Error reading time-series bounds: {e}")
        return {}

# Read experimental time-series data
experimental_data = read_experimental_data('Fig_a_Puretzki_data_points.xlsx')

# Calculate terminal CNT lengths for both methods
print("Calculating terminal CNT lengths for temperature range 500-1000°C...")

# Create 50 temperature points from 500 to 1000°C
temp_range = np.linspace(525, 900, 25)
terminal_lengths_puretzky = []
terminal_lengths_ma = []
terminal_lengths_ma_dp_sweep = []  # list of arrays: one per dp in dp_sweep_m
terminal_lengths_puretzky_dp_sweep = []  # list of arrays: one per dp in dp_sweep_m

time_span = [0, 500]  # seconds

initial_conditions = [0, 0, 0, 0, 0]

# Initialize Ma model (single dotted curve at dp=15 nm)
ma_model = MaGrowthModel(dp=dp_ma_line, P_FEEDSTOCK=P_FEEDSTOCK)
ma_models_dp_sweep = [MaGrowthModel(dp=_dp, P_FEEDSTOCK=P_FEEDSTOCK) for _dp in dp_sweep_m]

# Pre-allocate dp-sweep storage
terminal_lengths_ma_dp_sweep = [[] for _ in dp_sweep_m]
terminal_lengths_puretzky_dp_sweep = [[] for _ in dp_sweep_m]

for i, temp_C in enumerate(temp_range):
    print(f"Processing temperature {i+1}/25: {temp_C:.1f}°C")
    
    # Puretzky method
    try:
        solution = solve_ivp(
            lambda t, y: ode_system_puretzky(t, y, temp_C, dp),
            time_span,
            initial_conditions,
            method='BDF',
            rtol=1e-6,
            atol=1e-10
        )
        
        final_length_puretzky = calculate_length_puretzky(solution.y[4][-1], dp)
        terminal_lengths_puretzky.append(final_length_puretzky)
    except:
        terminal_lengths_puretzky.append(0)

    # Puretzky dp sweep for shaded band
    for j, _dp_m in enumerate(dp_sweep_m):
        try:
            sol_sweep = solve_ivp(
                lambda t, y: ode_system_puretzky(t, y, temp_C, _dp_m),
                time_span,
                initial_conditions,
                method='BDF',
                rtol=1e-6,
                atol=1e-10,
            )
            terminal_lengths_puretzky_dp_sweep[j].append(calculate_length_puretzky(sol_sweep.y[4][-1], _dp_m))
        except Exception:
            terminal_lengths_puretzky_dp_sweep[j].append(np.nan)
    
    # Ma method
    try:
        temp_K = temp_C + 273.15
        # Geometry-based uniform-zone time is used inside the model (so we don't force 5000 s here).
        final_length_ma = ma_model.calculate_terminal_length(temp_K, time_duration=None)
        terminal_lengths_ma.append(final_length_ma)
    except Exception as e:
        # IMPORTANT: don't append 0 on semilog-y plots (0 can't be drawn on a log axis).
        # Use NaN so matplotlib will skip those points, and print the error once to help debugging.
        if i == 0:
            print(f"[MaGrowthModel] WARNING: failed at T={temp_C:.1f}°C ({temp_K:.2f} K): {e}")
        terminal_lengths_ma.append(np.nan)

    # Ma method dp sweep for shaded band
    for j, _m in enumerate(ma_models_dp_sweep):
        try:
            temp_K = temp_C + 273.15
            terminal_lengths_ma_dp_sweep[j].append(_m.calculate_terminal_length(temp_K, time_duration=None))
        except Exception:
            terminal_lengths_ma_dp_sweep[j].append(np.nan)

# Convert to numpy arrays
terminal_lengths_puretzky = np.array(terminal_lengths_puretzky)
terminal_lengths_ma = np.array(terminal_lengths_ma)
terminal_lengths_ma_dp_sweep = np.array(terminal_lengths_ma_dp_sweep, dtype=float)  # shape: (n_dp, n_T)
terminal_lengths_puretzky_dp_sweep = np.array(terminal_lengths_puretzky_dp_sweep, dtype=float)  # shape: (n_dp, n_T)

# Quick sanity print for Puretzky dp band visibility
_p_pos = terminal_lengths_puretzky_dp_sweep[np.isfinite(terminal_lengths_puretzky_dp_sweep) & (terminal_lengths_puretzky_dp_sweep > 0)]
if _p_pos.size > 0:
    print(f"[Puretzky] dp-band terminal length range: min={_p_pos.min():.3e} µm, max={_p_pos.max():.3e} µm")
    _p_lo_dbg = np.nanmin(terminal_lengths_puretzky_dp_sweep, axis=0)
    _p_hi_dbg = np.nanmax(terminal_lengths_puretzky_dp_sweep, axis=0)
    _p_dbg_mask = np.isfinite(_p_lo_dbg) & np.isfinite(_p_hi_dbg) & (_p_lo_dbg > 0) & (_p_hi_dbg > 0)
    if np.any(_p_dbg_mask):
        _ratio = _p_hi_dbg[_p_dbg_mask] / _p_lo_dbg[_p_dbg_mask]
        print(f"[Puretzky] dp-band thickness: max(hi/lo)={float(np.nanmax(_ratio)):.6g}")
    _counts = []
    for _j, _dp_nm in enumerate(dp_sweep_nm.tolist()):
        _arr = terminal_lengths_puretzky_dp_sweep[_j]
        _counts.append(int(np.sum(np.isfinite(_arr) & (_arr > 0))))
    print(f"[Puretzky] dp-sweep valid counts by dp_nm={dp_sweep_nm.tolist()}: {_counts}")
else:
    print("[Puretzky] WARNING: no positive dp-band terminal lengths computed (shaded band will not appear).")

# Quick sanity print so it's obvious whether the green curve is out of view or invalid
pos_ma = terminal_lengths_ma[np.isfinite(terminal_lengths_ma) & (terminal_lengths_ma > 0)]
if len(pos_ma) > 0:
    print(f"[MaGrowthModel] terminal length range: min={pos_ma.min():.3e} µm, max={pos_ma.max():.3e} µm")
else:
    print("[MaGrowthModel] WARNING: no positive terminal lengths computed (curve will not appear on semilog-y plot).")

# Experimental data
exp_temp_C = [
    535.1926833428049, 543.9956777730257, 553.9006590223853, 563.8045067575024,
    569.3070474154217, 575.510739987853, 598.736929061133, 600.6984095413852,
    698.086809922933, 850.3450532974775, 874.6531988965412, 899.6926178590227
]  # Temperature in Celsius
exp_temp_K = [t + 273.15 for t in exp_temp_C]  # Convert to Kelvin
exp_lengths =  [
    0.39457199663712594, 0.6895947974780793, 1.0771624549883863, 1.912334446566704,
    2.5135466305017804, 1.3484727850043852, 9.176927054208775, 5.7359891880140035,
    152.1281109808873, 2.647791553562692, 1.546618049423535, 1.2283194699535995]  # Length in micrometers

# FIRST PLOT: Create comparison plot with shaded bounds area
plt.figure(figsize=(12, 12*6./8.18))

# Set font to Calibri (or fall back to default sans-serif if not available)
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['font.size'] = 24

# Convert temperature range to Kelvin for x-axis
temp_range_K = temp_range + 273.15

# Add shaded band for Puretzky from dp sweep (min/max across dp)
if terminal_lengths_puretzky_dp_sweep.size > 0:
    _p_lo = np.nanmin(terminal_lengths_puretzky_dp_sweep, axis=0)
    _p_hi = np.nanmax(terminal_lengths_puretzky_dp_sweep, axis=0)
    _p_mask = (
        np.isfinite(_p_lo)
        & np.isfinite(_p_hi)
        & np.isfinite(temp_range_K)
        & (_p_lo > 0.0)
        & (_p_hi > 0.0)
    )
    if np.any(_p_mask):
        plt.fill_between(
            temp_range_K[_p_mask],
            _p_lo[_p_mask],
            _p_hi[_p_mask],
            color="#15C5C5",
            alpha=0.30,
            linewidth=0,
            zorder=1,
            label=f"Puretzky dp band ({dp_sweep_nm.min():.0f}–{dp_sweep_nm.max():.0f} nm, Δ=3 nm)",
        )
        # Outline the envelope so thin bands are still visible
        plt.semilogy(
            temp_range_K[_p_mask],
            _p_lo[_p_mask],
            color="#15C5C5",
            linewidth=2,
            alpha=0.8,
            zorder=2,
            label="_nolegend_",
        )
        plt.semilogy(
            temp_range_K[_p_mask],
            _p_hi[_p_mask],
            color="#15C5C5",
            linewidth=2,
            alpha=0.8,
            zorder=2,
            label="_nolegend_",
        )

# Add shaded band for Ma method from dp sweep (min/max across dp)
if terminal_lengths_ma_dp_sweep.size > 0:
    _ma_lo = np.nanmin(terminal_lengths_ma_dp_sweep, axis=0)
    _ma_hi = np.nanmax(terminal_lengths_ma_dp_sweep, axis=0)
    # semilogy requires positive y
    _ma_mask = (
        np.isfinite(_ma_lo)
        & np.isfinite(_ma_hi)
        & np.isfinite(temp_range_K)
        & (_ma_lo > 0.0)
        & (_ma_hi > 0.0)
    )
    if np.any(_ma_mask):
        plt.fill_between(
            temp_range_K[_ma_mask],
            _ma_lo[_ma_mask],
            _ma_hi[_ma_mask],
            color="forestgreen",
            alpha=0.12,
            linewidth=0,
            label=f"Ma dp band ({dp_sweep_nm.min():.0f}–{dp_sweep_nm.max():.0f} nm, Δ=3 nm)",
        )

# Plot both methods with specified colors and line styles
plt.semilogy(temp_range_K, terminal_lengths_puretzky, '--', linewidth=4, 
             color='#15C5C5', label='Puretzky et al. (2005)', alpha=1)
plt.semilogy(temp_range_K, terminal_lengths_ma, ':', linewidth=4, 
             color='forestgreen', label='Ma et al. (2010)', alpha=1)

# Plot experimental data
plt.semilogy(exp_temp_K, exp_lengths, 'o', markersize=18,  markerfacecolor='none',markeredgewidth=3,
             color='#154357', label='Experimental Data', alpha=1)

plt.xlabel("Temperature [K]", fontsize=34, fontfamily='Calibri')
plt.ylabel("Terminal CNT Length [μm]", fontsize=34, fontfamily='Calibri')

# Remove grid
plt.grid(False)
plt.box(True)
# Set x and y limits
plt.xlim(775, 1400)  # Converted from 500-900°C to Kelvin
# Auto-adjust y-min so very small (but positive) model outputs are visible.
_ymin = 0.01
_ma_pos = terminal_lengths_ma[np.isfinite(terminal_lengths_ma) & (terminal_lengths_ma > 0)]
if len(_ma_pos) > 0:
    _ymin = min(_ymin, float(_ma_pos.min()) * 0.5)
# Also consider the Puretzky dp-band minimum (so it doesn't get clipped)
if isinstance(terminal_lengths_puretzky_dp_sweep, np.ndarray) and terminal_lengths_puretzky_dp_sweep.size > 0:
    _pband_pos = terminal_lengths_puretzky_dp_sweep[
        np.isfinite(terminal_lengths_puretzky_dp_sweep) & (terminal_lengths_puretzky_dp_sweep > 0)
    ]
    if _pband_pos.size > 0:
        _ymin = min(_ymin, float(np.min(_pband_pos)) * 0.5)
# Also consider the Ma dp-band minimum (so the shaded region doesn't get clipped)
if isinstance(terminal_lengths_ma_dp_sweep, np.ndarray) and terminal_lengths_ma_dp_sweep.size > 0:
    _band_pos = terminal_lengths_ma_dp_sweep[np.isfinite(terminal_lengths_ma_dp_sweep) & (terminal_lengths_ma_dp_sweep > 0)]
    if _band_pos.size > 0:
        _ymin = min(_ymin, float(np.min(_band_pos)) * 0.5)
_ymin = max(_ymin, 1e-12)
_ymax = 1_000_000.0
_y_candidates = []
_y_candidates.append(terminal_lengths_puretzky)
_y_candidates.append(terminal_lengths_ma)
if isinstance(terminal_lengths_puretzky_dp_sweep, np.ndarray) and terminal_lengths_puretzky_dp_sweep.size > 0:
    _y_candidates.append(terminal_lengths_puretzky_dp_sweep.ravel())
if isinstance(terminal_lengths_ma_dp_sweep, np.ndarray) and terminal_lengths_ma_dp_sweep.size > 0:
    _y_candidates.append(terminal_lengths_ma_dp_sweep.ravel())
try:
    _y_candidates.append(np.asarray(exp_lengths, dtype=float))
except Exception:
    pass
_y_all = np.concatenate([np.asarray(a, dtype=float).ravel() for a in _y_candidates if a is not None and np.size(a) > 0])
_y_all = _y_all[np.isfinite(_y_all) & (_y_all > 0)]
if _y_all.size > 0:
    _ymax = max(_ymax, float(np.max(_y_all)) * 1.2)
plt.ylim(_ymin, _ymax)

# Customize tick parameters
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='x', which='major', direction='out', bottom=True, top=False, width=2, length=8)
ax.tick_params(axis='y', which='major', direction='out', left=True, right=False, width=2,length=8)
ax.tick_params(axis='x', which='minor', direction='out', bottom=True, top=False, width=2)
ax.tick_params(axis='y', which='minor', direction='out', left=True, right=False, width=2)
ax.tick_params(axis='x', which='minor', direction='out', bottom=True, top=False, width=2, length=5)
ax.tick_params(axis='y', which='minor', direction='out', left=True, right=False, width=2, length=5)

# Remove top and right spines to clean up appearance
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

ax.tick_params(axis='both', which='major', labelsize=34)

# Legend
plt.legend(fontsize=22, loc='best')



plt.tight_layout()
# Save figures next to this script
save_directory = str(Path(__file__).resolve().parent)

# Create directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# For the first plot - replace your existing savefig commands with:
plt.savefig(os.path.join(save_directory, "cnt_growth_comparison_with_bounds.jpg"), format='jpg', dpi=300)
plt.savefig(os.path.join(save_directory, "cnt_growth_comparison_with_bounds.svg"), format='svg')
plt.savefig(os.path.join(save_directory, "cnt_growth_comparison_with_bounds.tiff"), format='tiff', dpi=300)
plt.savefig(os.path.join(save_directory, "cnt_growth_comparison_with_bounds.pdf"), format='pdf')



plt.show()










# SECOND PLOT: Time series with shaded bounds
selected_temps = [575, 600, 650, 700]

# Styling
plt.figure(figsize=(12, 12*6./8.18))
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['font.size'] = 24

# Define custom color and line style lists
#colors = ['#002060', '#0FDDF9', '#724804', '#F0EA00']
#colors = ['#002060', '#15C5C5', '#724804', '#F0EA00']
#colors = ['darkred', 'firebrick', 'indianred', 'lightcoral']
colors = ['#420E0A', '#6F3B79', 'firebrick', 'goldenrod']
linestyles = [':','-.','--','-']
symbols = ['^', 's', 'v', 'o']  # triangle_up, square, triangle_down, circle

for idx, temp_C in enumerate(selected_temps):
    temp_K = temp_C + 273.15
    color = colors[idx]
    ls = linestyles[idx]
    symbol = symbols[idx]

    # Puretzky method
    time_points = np.linspace(0, 500, 1000)
    solution = solve_ivp(
        lambda t, y: ode_system_puretzky(t, y, temp_C, dp),
        [0, 500],
        initial_conditions,
        method='BDF',
        rtol=1e-6,
        atol=1e-10,
        dense_output=True
    )

    y_dense = solution.sol(time_points)
    length_puretzky = calculate_length_puretzky(y_dense[4], dp)

    # Plot Puretzky method
    if ls == '-.':
        # Custom dash-dotted line
        line1, = plt.plot(time_points, length_puretzky, linewidth=4,
                          label=f"Puretzky {temp_C}°C", color=color, alpha=1)
        line1.set_dashes([4, 1, 1.5, 1])  # dash, gap, dot, gap
    else:
        # Default line style
        plt.plot(time_points, length_puretzky, linestyle=ls, linewidth=4,
                 label=f"Puretzky {temp_C}°C", color=color, alpha=1)

    # Ma method (time series) (no shaded regions requested for this figure)
    try:
        ma_t, ma_L = ma_model.calculate_growth_over_time(temp_K, [0, 500])
        plt.plot(
            ma_t,
            ma_L,
            linestyle=":",
            linewidth=3,
            color=color,
            alpha=0.9,
            label=f"Ma {temp_C}°C",
        )
    except Exception as e:
        if idx == 0:
            print(f"[MaGrowthModel] WARNING: time-series failed at T={temp_C:.1f}°C ({temp_K:.2f} K): {e}")

    # Add experimental data points for this temperature (if available)
    if temp_C in experimental_data and len(experimental_data[temp_C]['times']) > 0:
        plt.plot(experimental_data[temp_C]['times'], experimental_data[temp_C]['lengths'], 
                symbol, markersize=22, markerfacecolor='none', markeredgewidth=2,
                color=color, alpha=1, label=f"Exp {temp_C}°C")












# Axis labels
plt.xlabel("Time (s)", fontsize=34, fontfamily='Calibri')
plt.ylabel("CNT Length [μm]", fontsize=34, fontfamily='Calibri')

# Ticks and spines (same as first plot)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=34)
ax.tick_params(axis='x', which='major', direction='out', bottom=True, top=False, width=2, length=8)
ax.tick_params(axis='y', which='major', direction='out', left=True, right=False, width=2, length=8)
ax.tick_params(axis='x', which='minor', direction='out', bottom=True, top=False, width=2, length=5)
ax.tick_params(axis='y', which='minor', direction='out', left=True, right=False, width=2, length=5)

# Spine width
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

# No grid
plt.grid(False)
plt.box(True)

# Axis limits
plt.xlim(0, 500)
plt.ylim(0.1, 25)

# Legend
#lt.legend(fontsize=18, loc='best', ncol=2)

plt.tight_layout()

os.makedirs(save_directory, exist_ok=True)

# For the first plot - replace your existing savefig commands with:
plt.savefig(os.path.join(save_directory, "cnt_growth_vs_time_with_bounds.jpg"), format='jpg', dpi=600)
plt.savefig(os.path.join(save_directory, "cnt_growth_vs_time_with_bounds.svg"), format='svg')
plt.savefig(os.path.join(save_directory, "cnt_growth_vs_time_with_bounds.tiff"), format='tiff', dpi=600)
plt.savefig(os.path.join(save_directory, "cnt_growth_vs_time_with_bounds.pdf"), format='pdf')


plt.show()


