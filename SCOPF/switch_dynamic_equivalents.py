import sys
sys.path.append("C:\\Program Files\\DIgSILENT\\PowerFactory 2022 SP4\\Python\\3.9")
import powerfactory as pf
app = pf.GetApplication()
app.Show()  # Show before project activation to see graphics
app.ActivateProject('0. North GB Test System 18.09.23.IntPrj')
app.Hide()

def find_by_loc_name(elements: list, loc_name: str):
    for element in elements:
        if element.loc_name == loc_name:
            return element
    raise ValueError(loc_name, "not found")

loads = app.GetCalcRelevantObjects("*.ElmLod")
loads_dynamic_equivalent = [load for load in loads if 'Slack load' in load.loc_name]
loads = [load for load in loads if not load.outserv]
loads = [load for load in loads if 'Slack load' not in load.loc_name]
loads = [load for load in loads if load.loc_name != 'HVDC NET EMBED' and load.loc_name != 'HVDC WC load']
loads = [load for load in loads if load.loc_name != 'HVDC NET IC']
loads = [load for load in loads if load.loc_name[:2] != 'H2' and load.loc_name[:4] != 'BESS']
loads = [load for load in loads if load.loc_name[-4:] != 'NGET']

tfos = app.GetCalcRelevantObjects("*.ElmTr2")
tfos_dynamic_equivalent = [tfo for tfo in tfos if 'Transformer CMPLDW' in tfo.loc_name]
tfos = None

buses = app.GetCalcRelevantObjects("*.ElmTerm")
buses_dynamic_equivalent = [bus for bus in buses if 'Load Bus' in bus.loc_name or 'System Bus' in bus.loc_name]
buses = None

breakers = app.GetCalcRelevantObjects("*.ElmCoup")
breakers_dynamic_equivalent = [breaker for breaker in breakers if 'Switch CMPLDW' in breaker.loc_name]
breakers = None

IBRs = app.GetCalcRelevantObjects("*.ElmGenstat")
IBRs_dynamic_equivalent = [ibr for ibr in IBRs if 'Motor' in ibr.loc_name or 'DER_legacy' in ibr.loc_name or 'DER_G99' in ibr.loc_name or 'Static Load' in ibr.loc_name]
IBRs = None

def activate_dynamic_equivalents():
    for load in loads_dynamic_equivalent:
        load.outserv = 0
    for tfo in tfos_dynamic_equivalent:
        tfo.outserv = 0
    for bus in buses_dynamic_equivalent:
        bus.outserv = 0
    for breaker in breakers_dynamic_equivalent:
        breaker.on_off = 1
    for ibr in IBRs_dynamic_equivalent:
        ibr.outserv = 0

    for load in loads:
        load.scale0 = 0

def activate_loads():
    for load in loads_dynamic_equivalent:
        load.outserv = 1
    for tfo in tfos_dynamic_equivalent:
        tfo.outserv = 1
    for bus in buses_dynamic_equivalent:
        bus.outserv = 1
    for breaker in breakers_dynamic_equivalent:
        breaker.on_off = 0
    for ibr in IBRs_dynamic_equivalent:
        ibr.outserv = 1

    for load in loads:
        load.scale0 = 1

if __name__ == '__main__':
    DYNAMIC_EQUIVALENTS = False
    if DYNAMIC_EQUIVALENTS:
        activate_dynamic_equivalents()
    else:
        activate_loads()
    app.Show()