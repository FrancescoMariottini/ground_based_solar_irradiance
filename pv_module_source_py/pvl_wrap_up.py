
import clear_sky as csky
import numpy as np
import os
import pandas as pd
import pvlib


# not need load in cksy
_LINKE_TURBIDITIES_PATH = os.path.join(os.path.dirname(os.getcwd())+r"/data/LinkeTurbidities.h5")
# used in clear_sky
_NUMTHREADS = 4

# used in clear_sky
ATMOS_REFRACT = 0.5667
DEGDT = - 0.0002677 #The temperature dependence of the energy bandgap at reference conditions in units of 1/K
DELTA_T = 67.0
EGREF = 1.12 #The energy bandgap at reference temperature in units of eV. 1.121 eV for crystalline silicon. EgRef must be >0.
# used in clear_sky
EPOCH_YEAR = 2014 #The year in which a day of year input will be calculated.
IRRAD_REF = 1000
IVCURVE_PNTS = 0 #400
K_BOLTZMANN = 1.381 / (10 ** 23) # J/K
#numthread included for completeness but may not be necessary depending on used function
# used in clear_sky
PRESSURE = 101325.
Q_ELECTRON = 1.602 / (10 ** 19) # C
SOLAR_CONSTANT = 1366.1
TEMP_REF = 25 #C°

#pvlib dictionary, function parameters to columns
# 27/7/23 to be checked what used, AOI useful for uncertainty later
PLD:dict = {
"aoi":"AngInc", #if calculated from pvsyst
"apparent_zenith":"apparent_zenith",
"airmass_absolute":"airmass_absolute",
"datetime":"datetime",
"dhi":"DiffHor", 
"dni":"BeamHor", 
"dni_extra":"dni_extra", 
"effective_irradiance":"g_cmp11_ppuk", #TBC
"ghi":"GlobHor", 
"poa_global": "g_cmp11_ppuk",
"poa_direct": "poa_direct",
"poa_diffuse": "poa_diffuse",
"solar_azimuth":"AzSol", #solar_azimuth
"solar_zenith":"solar_zenith",
"sky_diffuse":"DifITrp",
"temp_air":"T_Amb", 
"temp_cell":"temp_cell", #TArray in PVsyst 
"time":"datetime",
"wind_speed":"WindVel"
}

# methods include also model parameters
METHODS:dict = {
    "extraradiation": "spencer", #'pyephem', 'spencer', 'asce', 'nrel'
    "iv": "desoto", #pvsyst, #cecmod
    "relative_airmass": "kastenyoung1989", #['kastenyoung1989', 'kasten1966', 'simple', 'pickering2002', 'youngirvine1967', 'young1994',
    "temp_cell": "sapm",
    "total_irradiance": "haydavies" #model
}



# not completed
class pv_module_model:
    def __init__(self, module:dict, sapm_module:dict,
                  methods:dict=METHODS, pld:dict=PLD,
                  atmos_refract=ATMOS_REFRACT, 
                  dEgdT=DEGDT,
                  delta_t=DELTA_T, EgRef=EGREF, epoch_year=EPOCH_YEAR,
                  ivcurve_pnts=IVCURVE_PNTS, k_Boltzmann=K_BOLTZMANN, 
                  irrad_ref=IRRAD_REF, 
                  numthreads=_NUMTHREADS, pressure=PRESSURE, 
                  q_electron=Q_ELECTRON,  
                  solar_constant=SOLAR_CONSTANT, temp_ref=TEMP_REF):
            self.module = module
            self.sapm_module = sapm_module
            self.methods = methods
            self.pld = pld
            # constants import
            # used in csky
            self.atmos_refract = atmos_refract
            self.dEgdT = dEgdT
            self.delta_t = delta_t
            self.EgRef = EgRef
            self.epoch_year = epoch_year
            self.irrad_ref = irrad_ref
            self.temp_ref = temp_ref
            self.ivcurve_pnts = ivcurve_pnts
            self.k_Boltzmann = k_Boltzmann
            # used in csky
            self.numthreads = numthreads
            # used in csky
            self.pressure = pressure
            self.q_electron = q_electron
            self.solar_constant = solar_constant  
            # adding parameter to iv
            iv_keys = ["alpha_sc","I_L_ref", "I_o_ref", "R_sh_ref", "R_s"]
            self.iv_parameters = {key: self.module[key] for key in iv_keys}
            # 14/8/23 using gamma_ref (pvsyst) instead of gamma_r
            self.iv_parameters["a_ref"] = self.module["gamma_ref"] * self.module["N_s"] * self.k_Boltzmann * (self.temp_ref+ 273.15) / self.q_electron,
            self.iv_parameters["dEgdT"] = self.dEgdT
            self.iv_parameters["irrad_ref"] = self.irrad_ref
            self.iv_parameters["temp_ref"] = self.temp_ref
            self.iv_parameters["EgRef"] = self.EgRef


    # testing effective irradiance method
    # 18/8/23 to be reviewed
    def add_spectral_factor_sapm(self, df:pd.DataFrame):
        # sapm_spectral_loss v.0.9.3
        # spectrum.spectral_factor_sap v.0.10
        # https://pvlib-python.readthedocs.io/en/stable/_modules/pvlib/spectrum/mismatch.html#calc_spectral_mismatch_field

        am_coeff= [self.sapm_module['A4'], self.sapm_module['A3'], self.sapm_module['A2'], self.sapm_module['A1'],
                self.sapm_module['A0']]         

        df["spectral_loss"] = df["airmassabsolute"].apply(lambda x: np.polyval(am_coeff, df["airmassabsolute"]))
        # spectral_loss = np.where(np.isnan(spectral_loss), 0, spectral_loss)
        # spectral_loss = np.maximum(0, spectral_loss)
        return df
    

        # https://pvlib-python.readthedocs.io/en/stable/_modules/pvlib/pvsystem.html#sapm_effective_irradiance
        
        """def sapm_effective_irradiance(self, df:pd.DataFrame):
                                    
                                    F1 = sapm_spectral_loss(airmass_absolute, module)
                                    
                                    
                                    poa_direct, poa_diffuse, airmass_absolute, aoi,
                                module)

        def 
            F1 = sapm_spectral_loss(airmass_absolute, module)
        F2 = sapm_aoi_loss(aoi, module)

        E0 = reference_irradiance

        Ee = F1 * (poa_direct*F2 + module['FD']*poa_diffuse) / E0

        return Ee
        """


    def add_effective_irradiance(self, df:pd.DataFrame, suffix="") -> pd.DataFrame:
        """
        wrap-up function to calculate the SAPM effective irradiance using the SAPM spectral loss and SAPM angle of incidence loss functions.
        :param df:
        :pld: pvlib parameters -> columns
        :module: pvlib 
        :suffix: suffix
        """

        # sapm_coeff = ["FD","A0","A1","A2","A3","A4","B0","B1","B2","B3","B4","B5"]
        df["effective_irradiance"+suffix] = df.apply(lambda x: pvlib.pvsystem.sapm_effective_irradiance(               
                        # poa_direct previously calculated from beam
                        poa_direct = x[self.pld["poa_direct"]], #* np.cos(np.deg2rad(x["angleofincidence"])), # x['poa_direct'], # x["BeamTrp"]
                        poa_diffuse =  x["poa_diffuse"],  #x['poa_diffuse'], # x["DifITrp"] + x["DifSInc"] 
                        airmass_absolute = x["airmassabsolute"], #x["am_abs"],
                        aoi = x["angleofincidence"],   #x["AngInc"], #x["aoi"],
                        module = self.sapm_module)
                        if x[self.pld["poa_global"]+suffix] != 0 else 0,                        
                        axis=1)

        """args = {
        "poa_direct": srs[self.pld["poa_direct"]], # x['poa_direct'], # x["BeamTrp"]
        "poa_diffuse": srs[self.pld["poa_diffuse"]],  #x['poa_diffuse'], # x["DifITrp"] + x["DifSInc"] 
        "airmass_absolute": srs["airmass_absolute"], #x["am_abs"], #calculated no need pld
        "aoi": srs["angleofincidence"],   #x["AngInc"], #x["aoi"], #calculated no need pld
        "module": self.module
        }
        srs["sapm_effective_irradiance"] = pvlib.pvsystem.sapm_effective_irradiance(**args)            
        """
        return df
    def add_temp_cell(self, df:pd.DataFrame) -> pd.DataFrame:
        if self.methods["temp_cell"] == "sapm":
                temperature_model_parameters = self.module["sapm"]
                df["temp_cell"] = df.apply(lambda x: pvlib.temperature.sapm_cell(
                                poa_global=x[self.pld["poa_global"]], 
                                temp_air=x[self.pld["temp_air"]], 
                                wind_speed=x[self.pld["wind_speed"]], 
                                **temperature_model_parameters)
                                if ((x[self.pld["poa_global"]] == x[self.pld["poa_global"]]) &
                                     (x[self.pld["poa_global"]] > 0))
                                else  np.nan, 
                                axis=1)

                """if self.methods["temp_cell"] == "sapm":
                        args = {
                        "poa_global": srs[self.pld["poa_global"]], 
                        "temp_air": srs[self.pld["temp_air"]], 
                        "wind_speed": srs[self.pld["wind_speed"]] 
                        }
                        temperature_model_parameters = self.module["sapm"]

                        srs["temp_cell"] = pvlib.temperature.sapm_cell(**args, **temperature_model_parameters)"""
        return df

    def add_mpp(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        wrap-up function
        :param df:
        :param x: dataframe dictionary
        :param ps_prms: parameters dictionary 
        :param calcparams: "desoto" or "pvsyst"
        :param calcmp: "singlediode", "bishop88"
        :param method: 'lambertw' #, 'newton', or 'brentq'
        :param ivcurve_pnts: 
        :param cecmodule:
        :param k_Boltzmann:
        :param t_ref: in C° transformed later
        :param q_electron:
        :return: add columns to df (see below). i & v as array for each point.
        """ 
        # no need to order
        clms_sd = ["photocurrent","saturation_current","resistance_series","resistance_shunt","nNsVth"]
        clms = ["i_sc","v_oc","i_mp","v_mp","p_mp","i_x","i_xx"]
        if self.ivcurve_pnts > 0: clms = clms + ["v","i"]
        calcparams = self.methods["iv"]

        if calcparams == "desoto":
            df[clms_sd] = df.apply(lambda x: pvlib.pvsystem.calcparams_desoto(
            effective_irradiance= x["effective_irradiance"], 
            temp_cell= x[self.pld["temp_cell"]], 
            alpha_sc= self.module["alpha_sc"], #The short-circuit current temperature coefficient of the module in units of A/C.
            a_ref= self.module["gamma_ref"] * self.module["N_s"] * self.k_Boltzmann * (self.temp_ref+ 273.15) / self.q_electron,
            # The product of the usual diode ideality factor (n, unitless), number of cells in series (Ns), and cell thermal voltage at reference conditions, in units of V        
            # {\displaystyle V_{\text{T}}=kT/q,} the thermal voltage.
            # gamma_ref = module["gamma_ref"], #The diode ideality factor
            # mu_gamma = module["mu_gamma"], #The temperature coefficient for the diode ideality factor, 1/K
            I_L_ref= self.module["I_L_ref"], #The light-generated current (or photocurrent) at reference conditions, in amperes.
            I_o_ref= self.module["I_o_ref"], #The dark or diode reverse saturation current at reference conditions, in amperes.
            R_sh_ref= self.module["R_sh_ref"], #The shunt resistance at reference conditions, in ohms.
            # R_sh_0 = module["R_sh_0"], #The shunt resistance at zero irradiance conditions, in ohms.
            R_s= self.module["R_s"], #The series resistance at reference conditions, in ohms.
            # cells_in_series = module["cells_in_series"], #The number of cells connected in series.
            #R_sh_exp=module["R_sh_exp"], #The exponent in the equation for shunt resistance, unitless. Defaults to 5.5.
            dEgdT= self.dEgdT, #The temperature dependence of the energy bandgap at reference conditions in units of 1/K
            EgRef= self.EgRef, #The energy bandgap at reference temperature in units of eV. 1.121 eV for crystalline silicon. EgRef must be >0.
            irrad_ref= self.irrad_ref, # Reference irradiance in W/m^2.
            temp_ref= self.temp_ref # Reference cell temperature in C.
            )
            if ((x["effective_irradiance"] > 0) & (x["effective_irradiance"] == x["effective_irradiance"])) else  
            pd.Series(dict(zip(clms_sd,[np.nan]*len(clms_sd))))          
            , axis=1, result_type='expand')

        df[clms] = df.apply(lambda x: pvlib.pvsystem.singlediode(
        photocurrent= x["photocurrent"],
        saturation_current= x["saturation_current"],
        resistance_series= x["resistance_series"],
        resistance_shunt= x["resistance_shunt"],
        nNsVth= x["nNsVth"], # k_Boltzmann * x[pld["temp_cell] / q_electron
        ivcurve_pnts= self.ivcurve_pnts,
        method= "lambertw")
        if ((x["effective_irradiance"] > 0) & (x["effective_irradiance"] == x["effective_irradiance"])) else
        pd.Series(dict(zip(clms,[np.nan]*len(clms))))
        , axis=1, result_type='expand')
  
        # not clear why array returned
        df["i_mp"]=df["i_mp"].apply(float)
        df["v_mp"]=df["v_mp"].apply(float)
        return df
            
    def get_mp(self, df:pd.DataFrame, temp_calc=False, diffuse_calc=False) -> pd.DataFrame:
          if diffuse_calc:
            def calculate_diffuse(x):
                    d = max(0, x[self.pld["poa_global"]]-x[self.pld["poa_direct"]]*np.cos(np.deg2rad(x["angleofincidence"])))
                    return d          
            # sapm_spectral_loss v.0.9.3
            # spectrum.spectral_factor_sap v.0.10
            # https://pvlib-python.readthedocs.io/en/stable/_modules/pvlib/spectrum/mismatch.html#calc_spectral_mismatch_field
            df["poa_diffuse"] = df.apply(lambda x: calculate_diffuse(x), axis=1)
          df = self.add_effective_irradiance(df)
          if temp_calc: df = self.add_temp_cell(df)
          else: df["temp_cell"] = df[self.pld["temp_cell"]]
          df = self.add_mpp(df)
          return df
          

        
def pvl_azm_from_pvs(pvs_azm):
    """
    Azimutal conversion for pvlib when using pvsyst convention
    """ 
    #24/1/23 updated
    return (pvs_azm+540)%360


