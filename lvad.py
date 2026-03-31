import math
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from IPython.display import display

def smooth_param(t, rest_val, ex_val, t_mid_up, t_mid_down, k):
    """Sigmoid-blended parameter between rest and exercise phases."""
    alpha_up   = 1.0 / (1.0 + math.exp(-k * (t - t_mid_up)))
    alpha_down = 1.0 / (1.0 + math.exp(-k * (t - t_mid_down)))
    blend = alpha_up - alpha_down          # 0 → 1 → 0 over the exercise window
    return rest_val + (ex_val - rest_val) * blend


def target(x1: float, y1: float, CL: np.ndarray, Plved_grid: np.ndarray, SF: float = 1.0):
    """Immediate Starling-like target."""
    ControlLine = CL * SF if SF != 1.0 else CL
    slope = -1.96
    y = slope * (Plved_grid - x1) + y1

    # Find the closest intersection robustly
    diff = np.abs(y - ControlLine)
    idx = np.argmin(diff)

    # Fallback in case the lines completely miss each other
    if diff[idx] > 1.5:
        return np.nan, np.nan

    return float(Plved_grid[idx]), float(y[idx])

class CardioModel:
    def __init__(self, patient_condition:str='healthy', lvad_add:str='no', omega:float=0.0,
                 controller1:str='no',controller2:str='no', exercise:str='no',
                 Kp:float=0.0, Ki:float=0.0,Kd:float=0.0, Kp2:float=0.0, Ki2:float=0.0, Kd2:float=0.0):
        self.patient_condition = patient_condition
        self.lvad_add = lvad_add
        self.omega = omega
        self.omega_baseline = omega
        self.controller1 = controller1
        self.controller2 = controller2
        self.exercise = exercise
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Kp2 = Kp2
        self.Ki2 = Ki2
        self.Kd2 = Kd2            # <-- Store Kd2
        self.previous_error1 = 0.0
        self.previous_error2 = 0.0
        self.rpm_history = []

        pressures = []
        volumes = []
        flows = []
        Elastances = []
        baroreflex = []
        f_es_list = []
        f_ev_list = []


        # Resistances (mmHg·s/mL)
        R_ao = 0.0398
        R_svr0 = 0.62
        R_sv = 0.023
        R_pa = 0.005
        R_pvr = 0.12
        R_pv = 0.011
        R_AV = 0.008
        R_MV = 0.005
        R_PV = 0.003
        R_TV = 0.001

        # Compliance
        C_ao = 0.08
        C_sar = 1.33
        C_sv = 20.5
        C_pa = 0.18
        C_par = 3.0
        C_pv = 20.5

        # Inertances (mmHg·s^2/mL)
        L_ao = 0.0005
        L_pa = 0.00005

        # Unstressed volumes V0 (mL)
        V0_lv = 5
        V0_la = 4
        V0_rv = 10
        V0_ra = 4
        V0_ao = 55
        V0_sar = 0
        V0_sv = 2500
        V0_pa = 100
        V0_par = 0
        V0_pv = 150

        # Elastance parameters (mmHg/mL)
        E_lv_min = 0.04
        E_rv_min = 0.04
        E_la_min = 0.15
        E_la_max = 0.25
        E_ra_min = 0.15
        E_ra_max = 0.25
        E_lv_max0 = 1.8
        E_rv_max0 = 0.6

        # Baroreflex parameters
        tau_p = 2.076
        tau_z = 6.37
        f_min = 2.52
        f_max = 47.78
        p_n = 92
        k_a = 11.758

        f_es_inf = 2.1
        f_es_0 = 16.11
        f_ev_inf = 6.3
        f_ev_0 = 3.2
        f_as_0 = 25

        k_es = 0.0675
        k_ev = 7.06
        G_R_svr = 0.36
        tau_R_svr = 6.0
        D_R_svr = 2.0
        f_es_min  = 2.66
        G_E_lv_max = 0.475
        tau_E_lv_max = 8.0
        D_E_lv_max = 2.0

        G_E_rv_max = 0.282
        tau_E_rv_max = 8.0
        D_E_rv_max = 2.0

        G_T_s = -0.13
        tau_T_s = 2.0
        D_T_s = 2.0

        G_T_v = 0.09
        tau_T_v = 1.5
        D_T_v = 0.2
        P = 0.0
        T0 = 0.58 # initial heart period (s)

        # Initial values for baroreflex state variables
        del_R_svr = 0.0
        del_E_lv_max = 0.0
        del_E_rv_max = 0.0
        del_T_sym = 0.0
        del_T_vag = 0.0

        # Initialize controlled parameters at baseline
        R_svr = R_svr0
        E_lv_max = E_lv_max0
        E_rv_max = E_rv_max0
        T = T0

        #----------------------------------------------------------------------
        if self.patient_condition == 'lv_failure':
            C_sar = 1.02
            R_pv =  0.027
            k_a = 14
            G_R_svr = 0.3
            G_E_lv_max = 0.15
            G_E_rv_max = 0.245
            G_T_s = -0.07
            G_T_v = 0.07

            E_lv_max0 = 0.05
            R_svr0 =    0.71
            T0 =        0.52
            R_pvr =     0.14
        #----------------------------------------------------------------------

        #LVAD parameters (HeartMate III) from Liu
        if self.lvad_add == 'yes':
            beta_a = -1.8e-3
            beta_b = -1.2e-5
            beta_c = 7.3e-6
            R_i   = 0.0677
            L_i   = 0.0127
            alpha = -3.5
            P_th = 1.0
            omega = self.omega
            Q_irp = 0.0

        # Initial volumes (mL) for each compartment (with initial pressure assumptions)
        P_ao_init = 95
        P_sa_init = 95
        P_sv_init = 5
        P_pa_init = 15
        P_par_init = 10
        P_pv_init = 5

        Vlv = V0_lv + 135
        Vrv = V0_rv + 130
        Vla = V0_la + 30
        Vra = V0_ra + 30
        Vao = V0_ao + C_ao * P_ao_init
        Vsar = V0_sar + C_sar * P_sa_init
        Vsv = V0_sv + C_sv * P_sv_init
        Vpa = V0_pa + C_pa * P_pa_init
        Vpar = V0_par + C_par * P_par_init
        Vpv = V0_pv + C_pv * P_pv_init

        # Initial flows
        Q_ao = 0.0
        Q_pa = 0.0
        t_cycle = 0.0
        dt = 0.001
        t_max = 600.0
        steps = int(t_max / dt)
        stepdt = []

        cycle_sum = 0.0
        cycle_count = 0
        error_integral = 0.0
        target_MAP = 86.0
        omega_min = 3000.0
        omega_max = 4700.0
        Pao_average_list = []
        Q_irp_average_list=[]
        Q_irp_target_list=[]
        LVEDP=[]
        LVEDV=[]

        cycle_sum_Pao = 0.0
        cycle_sum_Qirp = 0.0
        cycle_count = 0

        cycle_max_Vlv = 0.0  # Track max volume (EDV)
        current_LVEDP = 0.0  # Track corresponding pressure

        PI_ao_list = []
        PI_irp_list = []
        cycle_sum_Qao = 0.0

        cycle_max_Qao = -float('inf')
        cycle_min_Qao = float('inf')

        cycle_max_Qirp = -float('inf')
        cycle_min_Qirp = float('inf')

        Plved_CL = np.arange(0.0, 50.0 + 0.0001, 0.0001)
        ControlLine = 10.3 + (-10.3 / (1.0 + (Plved_CL / 7.0) ** 2.3))
        # -----------------------
        # Time-stepping simulation
        # -----------------------
        # ── Sigmoid transition settings ──────────────────────────────────────────────
        t_mid_up   = t_max * 0.42   # centre of rest→exercise ramp  (tune as needed)
        t_mid_down = t_max * 0.68   # centre of exercise→rest ramp
        k_sigmoid  = 0.15           # steepness (higher = sharper; lower = smoother)

        # ── Rest values (healthy) ────────────────────────────────────────────────────
        rest_h = dict(
            R_svr0=0.62, E_lv_max0=1.8, E_rv_max0=0.6, T0=0.58,
            E_ra_max=0.25, E_la_max=0.25, R_pvr=0.12,
            C_sv=20.5, V0_sv=2500.0, V0_pv=150.0,
            R_sv=0.023, R_pv=0.011, R_MV=0.005, E_lv_min=0.04,
            target_MAP=86.0
        )
        # ── Exercise values (healthy) ────────────────────────────────────────────────
        ex_h = dict(
            R_svr0=0.62*0.5, E_lv_max0=1.8*2.0, E_rv_max0=0.6*2.0, T0=0.58*0.7,
            E_ra_max=0.25*1.5, E_la_max=0.25*1.5, R_pvr=0.12*0.7,
            C_sv=20.5*0.8, V0_sv=2500*0.85, V0_pv=150*0.8,
            R_sv=0.023*0.8, R_pv=0.011, R_MV=0.005, E_lv_min=0.11,
            target_MAP=100.0
        )

        # ── Rest values (LV failure) ─────────────────────────────────────────────────
        rest_f = dict(
            R_svr0=0.71, E_lv_max0=0.05, E_rv_max0=0.6, T0=0.52,
            E_ra_max=0.25, E_la_max=0.25, R_pvr=0.14,
            C_sv=20.5, V0_sv=2500.0, V0_pv=150.0,
            R_sv=0.023, R_pv=0.011, R_MV=0.005, E_lv_min=0.04,
            target_MAP=85.0
        )
        # ── Exercise values (LV failure) ─────────────────────────────────────────────
        ex_f = dict(
            R_svr0=0.71*0.6, E_lv_max0=0.05*1.5, E_rv_max0=0.6*1.5, T0=0.52*0.7,
            E_ra_max=0.25*1.5, E_la_max=0.25*1.5, R_pvr=0.14*0.8,
            C_sv=20.5*0.8, V0_sv=2500*0.85, V0_pv=150*0.8,
            R_sv=0.023*0.8, R_pv=0.011, R_MV=0.005, E_lv_min=0.11,
            target_MAP=95.0
        )
        for step in range(steps):
            # Update cycle time
            t_cycle += dt
            if t_cycle >= T:
              current_LVEDP = PLV
              current_LVEDV = Vlv
              # Track maximum volume and corresponding pressure for the current cycle
              LVEDV.append(current_LVEDV)
              LVEDP.append(current_LVEDP)
              avg_Qao = cycle_sum_Qao / cycle_count if cycle_count > 0 else 0
              avg_Pao = cycle_sum_Pao / cycle_count if cycle_count > 0 else 0
              avg_Qirp = cycle_sum_Qirp / cycle_count if cycle_count > 0 else 0
              #Q_irp_average_list.append(avg_Qirp)
              Pao_average_list.append(avg_Pao)

              # Aortic PI (Normal Heart)
              if avg_Qao != 0:
                  pi_ao = (cycle_max_Qao - cycle_min_Qao) / avg_Qao
              else:
                  pi_ao = 0.0
              PI_ao_list.append(pi_ao)

              # Pump PI (LVAD)
              if self.lvad_add == 'yes':
                  if avg_Qirp != 0:
                      pi_irp = (cycle_max_Qirp - cycle_min_Qirp) / avg_Qirp
                  else:
                      pi_irp = 0.0
                  PI_irp_list.append(pi_irp)
              # --- Controller 1 (MAP Control) ---
              # --- Controller 1 (MAP Control) ---
              # --- Controller 1 (MAP Control) ---
              if self.controller1 == 'yes':
                  error = target_MAP - avg_Pao

                  # 1. Calculate Derivative
                  derivative = (error - self.previous_error1) / T

                  # 2. FIXED ANTI-WINDUP: Only freeze if pushing further into saturation
                  if (self.omega >= omega_max and error > 0) or (self.omega <= omega_min and error < 0):
                      pass # Saturated: do not wind up further
                  else:
                      error_integral += error * T

                  # 3. Update PID formula: P + I + D
                  domega = (self.Kp * error) + (self.Ki * error_integral) + (self.Kd * derivative)

                  # Apply change (keeping your existing slew rate limiter or direct assignment)
                  desired_omega = self.omega_baseline + domega
                  delta_omega = desired_omega - self.omega
                  max_delta_omega = 50.0

                  self.omega += max(-max_delta_omega, min(delta_omega, max_delta_omega))
                  self.omega = max(omega_min, min(self.omega, omega_max))

                  # 3. Store error for next beat
                  self.previous_error1 = error

              # --- Controller 2 (Starling-Like Control) ---
              # --- Controller 2 (Starling-Like Control) ---
              if self.controller2 == 'yes':
                  # 1. Convert flow from mL/s to L/min for the control target
                  avg_Qirp_Lmin = avg_Qirp * 60.0 / 1000.0

                  LVEDP_Target_current, Qirp_target_current_Lmin = target(current_LVEDP, avg_Qirp_Lmin, ControlLine, Plved_CL, 1.0)

                  if not np.isnan(Qirp_target_current_Lmin):
                      # 2. Calculate error in L/min!
                      Q_irp_average_list.append(avg_Qirp_Lmin)
                      Q_irp_target_list.append(Qirp_target_current_Lmin)
                      error = Qirp_target_current_Lmin - avg_Qirp_Lmin

                      if omega_min < self.omega < omega_max:
                          error_integral += error * T
                      derivative = (error - self.previous_error2) / T
# 6. Apply Slew Rate Limiter (Highly recommended to keep this)
                      domega = (Kp2 * error) + (Ki2 * error_integral) + (self.Kd2 * derivative)
                      desired_omega = self.omega_baseline + domega
                      max_delta_omega = 50.0
                      delta_omega = desired_omega - self.omega

                      self.omega += max(-max_delta_omega, min(delta_omega, max_delta_omega))
                      self.omega = max(omega_min, min(self.omega, omega_max)) # Keep in bounds

                      # 7. Store the current error for the NEXT beat's derivative calculation
                      self.previous_error2 = error

              # Reset cycle trackers for the next heartbeat!
              cycle_sum_Pao = 0.0
              cycle_sum_Qirp = 0.0
              cycle_count = 0
              #cycle_max_Vlv = 0.0 # Reset EDV tracker

              cycle_sum_Qao = 0.0
              cycle_max_Qao = -float('inf')
              cycle_min_Qao = float('inf')
              cycle_max_Qirp = -float('inf')
              cycle_min_Qirp = float('inf')

              t_cycle -= T
              self.rpm_history.append(self.omega)

            # Exercise conditions
            # Exercise conditions
            # ── Smooth exercise transition ────────────────────────────────────────────────
            if self.exercise == 'yes':
                t_now = step * dt

                if self.patient_condition == 'healthy':
                    rv, ev = rest_h, ex_h
                else:  # lv_failure
                    rv, ev = rest_f, ex_f

                def sp(key):
                    return smooth_param(t_now, rv[key], ev[key], t_mid_up, t_mid_down, k_sigmoid)

                R_svr0    = sp('R_svr0')
                E_lv_max0 = sp('E_lv_max0')
                E_rv_max0 = sp('E_rv_max0')
                T0        = sp('T0')
                E_ra_max  = sp('E_ra_max')
                E_la_max  = sp('E_la_max')
                R_pvr     = sp('R_pvr')
                C_sv      = sp('C_sv')
                V0_sv     = sp('V0_sv')
                V0_pv     = sp('V0_pv')
                R_sv      = sp('R_sv')
                R_pv      = sp('R_pv')
                R_MV      = sp('R_MV')
                E_lv_min  = sp('E_lv_min')
                target_MAP = sp('target_MAP')
# ─────────────────────────────────────────────────────────────────────────────
            
  


            t_n = t_cycle / (0.2 + 0.15 * T)
            E_n = 1.55 * ((t_n / 0.7) ** 1.9 / (1 + (t_n / 0.7) ** 1.9)) * (1 / (1 + (t_n / 1.17) ** 21.9))

            e_la = 0.0
            e_ra = 0.0
            if 0.9 * T <= t_cycle < 0.99 * T:
                phi = (t_cycle - 0.9 * T) / (0.09 * T)
                e_la = 1 - math.cos(2 * math.pi * phi)
                e_ra = e_la

            E_lv = (E_lv_max - E_lv_min) * E_n + E_lv_min
            E_rv = (E_rv_max - E_rv_min) * E_n + E_rv_min
            E_la = (E_la_max - E_la_min) / 2.0 * e_la + E_la_min
            E_ra = (E_ra_max - E_ra_min) / 2.0 * e_ra + E_ra_min

            PLV = E_lv * (Vlv - V0_lv)
            PRV = E_rv * (Vrv - V0_rv)
            PLA = E_la * (Vla - V0_la)
            PRA = E_ra * (Vra - V0_ra)
            Pao = (Vao - V0_ao) / C_ao
            Psar = (Vsar - V0_sar) / C_sar
            Psv = (Vsv - V0_sv) / C_sv
            Ppa = (Vpa - V0_pa) / C_pa
            Ppar = (Vpar - V0_par) / C_par
            Ppv = (Vpv - V0_pv) / C_pv

            Q_av = max(0.0, PLV - Pao) / R_AV
            Q_mv = max(0.0, PLA - PLV) / R_MV
            Q_pv_valve = max(0.0, PRV - Ppa) / R_PV
            Q_tv = max(0.0, PRA - PRV) / R_TV

            Q_ao += dt * ((Pao - R_ao * Q_ao) - Psar) / L_ao
            Q_pa += dt * ((Ppa - R_pa * Q_pa) - Ppar) / L_pa




            if self.lvad_add == 'yes':
                R_k = 0.0 if PLV > P_th else alpha * (PLV - P_th)
                H = beta_a * Q_irp**2 + beta_b * Q_irp * self.omega + beta_c * self.omega**2
                Q_irp += dt * ((PLV - Pao + H) - (R_i + R_k) * Q_irp) / L_i
            else:
              pass

            Q_svr = (Psar - Psv) / R_svr
            Q_pvr = (Ppar - Ppv) / R_pvr
            Q_sv = (Psv - PRA) / R_sv
            Q_pv_vein = (Ppv - PLA) / R_pv

            # Update volumes using flow continuity (V_new = V_old + dt * (inflow - outflow))
            if self.lvad_add == 'yes':
                Vlv += dt * (Q_mv - (Q_av + Q_irp))
                Vao += dt * ((Q_av + Q_irp) - Q_ao)
            else:
                Vlv += dt * (Q_mv - Q_av)
                Vao += dt * (Q_av - Q_ao)

            cycle_sum_Pao += Pao
            if self.lvad_add == 'yes':
              cycle_sum_Qirp+=Q_irp
            cycle_count += 1
            cycle_sum_Qao += Q_ao

            if Q_ao > cycle_max_Qao: cycle_max_Qao = Q_ao
            if Q_ao < cycle_min_Qao: cycle_min_Qao = Q_ao

            if self.lvad_add == 'yes':
                if Q_irp > cycle_max_Qirp: cycle_max_Qirp = Q_irp
                if Q_irp < cycle_min_Qirp: cycle_min_Qirp = Q_irp
            Vsar += dt * (Q_ao - Q_svr)
            Vsv += dt * (Q_svr - Q_sv)
            Vra += dt * (Q_sv - Q_tv)
            Vrv += dt * (Q_tv - Q_pv_valve)
            Vpa += dt * (Q_pv_valve - Q_pa)
            Vpar += dt * (Q_pa - Q_pvr)
            Vpv += dt * (Q_pvr - Q_pv_vein)
            Vla += dt * (Q_pv_vein - Q_mv)


            # Track maximum volume and corresponding pressure for the current cycle
            # if Vlv > cycle_max_Vlv:
            #     cycle_max_Vlv = Vlv
            #     current_LVEDP = PLV



            if step == 0:
                dPao_dt = 0.0
            else:
                dPao_dt = (Pao - pressures[-1][4]) / dt

            P += dt * ((tau_z * dPao_dt) + Pao - P) / tau_p

            f_as = (f_min + f_max * math.exp((P - p_n) / k_a)) / (1 + math.exp((P - p_n) / k_a))
            f_es = f_es_inf + (f_es_0 - f_es_inf) * math.exp(-k_es * f_as)
            f_ev = (f_ev_0 + f_ev_inf * math.exp((f_as - f_as_0) / k_ev)) / (1 + math.exp((f_as - f_as_0) / k_ev))

            f_es_list.append(f_es)
            f_ev_list.append(f_ev)
            stepdt.append(step * dt)

            if step * dt >= D_R_svr and f_es >= f_es_min:
                f_es_lagged = f_es_list[int(step * dt / dt) - int(D_R_svr / dt)]
                sigma_R = G_R_svr * math.log(f_es_lagged - f_es_min + 1)
            else:
                sigma_R = 0.0
            del_R_svr += dt * (sigma_R - del_R_svr) / tau_R_svr
            R_svr = R_svr0 + del_R_svr

            if step * dt >= D_E_lv_max and f_es >= f_es_min:
                f_es_lagged = f_es_list[int(step * dt / dt) - int(D_E_lv_max / dt)]
                sigma_E_lv = G_E_lv_max * math.log(f_es_lagged - f_es_min + 1)
            else:
                sigma_E_lv = 0.0
            del_E_lv_max += dt * (sigma_E_lv - del_E_lv_max) / tau_E_lv_max
            E_lv_max = E_lv_max0 + del_E_lv_max

            if step * dt >= D_E_rv_max and f_es >= f_es_min:
                f_es_lagged = f_es_list[int(step * dt / dt) - int(D_E_rv_max / dt)]
                sigma_E_rv = G_E_rv_max * math.log(f_es_lagged - f_es_min + 1)
            else:
                sigma_E_rv = 0.0
            del_E_rv_max += dt * (sigma_E_rv - del_E_rv_max) / tau_E_rv_max
            E_rv_max = E_rv_max0 + del_E_rv_max

            if step * dt >= D_T_s and f_es >= f_es_min:
                f_es_lagged = f_es_list[int(step * dt / dt) - int(D_T_s / dt)]
                sigma_T_sym = G_T_s * math.log(f_es_lagged - f_es_min + 1)
            else:
                sigma_T_sym = 0.0
            del_T_sym += dt * (sigma_T_sym - del_T_sym) / tau_T_s

            if step * dt >= D_T_v:
                f_ev_lagged = f_ev_list[int(step * dt / dt) - int(D_T_v / dt)]
                sigma_T_vag = G_T_v * f_ev_lagged
            else:
                sigma_T_vag = 0.0
            del_T_vag += dt * (sigma_T_vag - del_T_vag) / tau_T_v

            T = T0 + del_T_sym + del_T_vag
            if T < 0.2:
                T = 0.2


            pressures.append([PLV, PLA, PRV, PRA, Pao, Psar, Psv, Ppa, Ppar, Ppv])
            volumes.append([Vlv, Vla, Vrv, Vra, Vao, Vsar, Vsv, Vpa, Vpar, Vpv])


            if self.lvad_add == 'yes':
                flows.append([Q_av, Q_mv, Q_pv_valve, Q_tv, Q_ao, Q_svr, Q_sv, Q_pa, Q_pvr, Q_pv_vein, Q_irp])
            else:
                flows.append([Q_av, Q_mv, Q_pv_valve, Q_tv, Q_ao, Q_svr, Q_sv, Q_pa, Q_pvr, Q_pv_vein])
            Elastances.append([E_lv, E_la, E_rv, E_ra])
            baroreflex.append([R_svr, E_lv_max, E_rv_max, T])



        # ---------------------------------------------------------
        # POST-SIMULATION DATA EXTRACTION
        # ---------------------------------------------------------

        # Save results to object attributes first
        self.pressures = pressures
        self.volumes = volumes
        self.flows = flows
        self.Elastances = Elastances
        self.baroreflex = baroreflex
        self.f_es_list = f_es_list
        self.f_ev_list = f_ev_list
        self.stepdt = stepdt
        self.Pao_average_list = Pao_average_list
        self.time = [i * dt for i in range(steps)]
        self.heart_rate = 60 / T0
        self.LVEDP=LVEDP
        self.LVEDV=LVEDV
        self.Q_irp_average_list=Q_irp_average_list
        self.Q_irp_target_list=Q_irp_target_list
        self.PI_ao_list = PI_ao_list
        self.PI_irp_list = PI_irp_list
        # # Extract Left Ventricular Volume (Vlv) and Pressure (PLV) using numpy
        # vlv_data_full = np.array([v[0] for v in self.volumes])
        # plv_data_full = np.array([p[0] for p in self.pressures])

        # # 0.15 seconds ensures we don't skip peaks even at max simulated heart rate
        # safe_distance = 5000

        # peaks, _ = find_peaks(vlv_data_full, distance=safe_distance)

        # # Extract LVEDV and LVEDP values and save them as class attributes
        # self.LVEDV = vlv_data_full[peaks].tolist()
        # self.LVEDP = plv_data_full[peaks].tolist()


    # ---------------------------------------------------------
    # BONUS: New Class Method to generate your DataFrame Table
    # ---------------------------------------------------------
    def get_ed_table(self):
        """Returns and displays a Pandas DataFrame of beat-by-beat EDV and EDP."""
        if not hasattr(self, 'LVEDV') or not self.LVEDV:
            return "No data available. Ensure simulation ran successfully."

        beat_numbers = np.arange(1, len(self.LVEDV) + 1)

        ed_table = pd.DataFrame({
            'Beat Number': beat_numbers,
            'EDV (mL)': self.LVEDV,
            'EDP (mmHg)': self.LVEDP,
            'Native PI': self.PI_ao_list,                           # <--- ADDED
            'LVAD PI': self.PI_irp_list if self.PI_irp_list else 0  # <--- ADDED
        }).round(2)

        display(ed_table)
        return ed_table