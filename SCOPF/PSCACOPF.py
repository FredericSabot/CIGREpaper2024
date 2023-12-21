import sys
sys.path.append(r'C:\GAMS\36\apifiles\Python\api_39')
sys.path.append(r'C:\GAMS\36\apifiles\Python\gams')

sys.path.append("C:\\Program Files\\DIgSILENT\\PowerFactory 2022 SP4\\Python\\3.9")

import csv
import os
import numpy as np
import gams
import sys
import datetime
from cmath import pi
import pypowsybl as pp
from pathlib import Path
import shutil
import pandas as pd

import powerfactory as pf
app = pf.GetApplication()
app.Show()  # Show before project activation to see graphics
app.ActivateProject('0. North GB Test System.IntPrj')
app.Hide()  # Disable GUI during computations to avoid slowing down the script

def find_by_loc_name(elements: list, loc_name: str):
    for element in elements:
        if element.loc_name == loc_name:
            return element
    raise ValueError(loc_name, "not found")

def clearSimEvents():
    faultFolder = app.GetFromStudyCase("Simulation Events/Fault.IntEvt")
    cont = faultFolder.GetContents()
    for obj in cont:
        obj.Delete()

def add_short_circuit_event(obj, time, fault_type, position = 50):
    faultFolder = app.GetFromStudyCase("Simulation Events/Fault.IntEvt")
    event = faultFolder.CreateObject("EvtShc", obj.loc_name)
    event.p_target = obj
    event.time = time
    event.i_shc = fault_type  # 0 = 3 phase fault, 4 = faut clearing
    obj.ishclne = 1  # Set available for RMS simulation
    obj.fshcloc = position

def add_switch_event(obj, time, switch_action):
    faultFolder = app.GetFromStudyCase("Simulation Events/Fault.IntEvt")
    event = faultFolder.CreateObject("EvtSwitch", obj.loc_name)
    event.p_target = obj
    event.time = time
    event.i_switch = switch_action

def export_results_to_csv(output_csv):
    comres = app.GetFromStudyCase('ComRes')
    comres.iopt_csel = 0  # export only selected variables = 1, export all variables = 0
    comres.iopt_tsel = 0  # user defined interval, 0 = full duration
    comres.iopt_locn = 1  # first header = model name
    comres.ciopt_head = 1  # second header = parameter name
    comres.f_name = output_csv  # the name of the file where the results will be saved
    comres.iopt_exp = 6  # Export to 4=text file, 3=CFD file etc, 6 = csv
    comres.Execute()

def check_stability(results_csv, buses, sync_machines, t_end):
    stable = True
    with open(results_csv, "r") as f:
        data = pd.read_csv(f, sep=",", header=[0, 1])

    if data['All calculations', 'b:tnow in s'].iloc[-1] < t_end:
        raise RuntimeError('Convergence issue in dynamic simulation')

    # Check for voltage issues
    for bus in buses:
        if 'Scap' in bus.loc_name:
            continue  # Disregard 0 voltage at node between series capa and disconnected line
        try:
            bus_data = data[bus.loc_name]
        except KeyError:
            print('Missing data', bus.loc_name)
            continue
        u = bus_data['m:u1 in p.u.'].iloc[-1]  # Voltages at final point of simulation
        if u < 0.9 or u > 1.1:
            print('Voltage issue at bus', bus.loc_name, u)
            if u < 0.85 or u > 1.1:
                stable = False
                break

    """
    # Need to add outofstep parameter to generators manually first
    # Check for loss of synchronism of synchronous machines
    for gen in sync_machines:
        if bus_data[gen.loc_name]['s:outofstep']:
            stable = False
            print('Generator', gen.loc_name, 'lost synchronism')
            break
    """

    return stable

baseMVA = 100  # Sbase in Powerfactory
year = 2031
print('Study Year: %s' %year)

# TODO: define scenarios
scenario = 1
if scenario == 1:  # Summer minimum PM Leading the way
    SCOTLAND_WIND_AVAILABILITY = 0.8
    NGET_WIND_AVAILABILITY = 0.8 * 0.58
    SCENARIO_NAME = 'SummerPM_{}_leading'.format(year)
    SOLAR_FACTOR = 0.68  # All-time peak according to https://www.solar.sheffield.ac.uk/pvlive/ (checked on 14/12/2023, peak reached on 2023-04-20 12:30PM which is not really in the summer)
    CHP_FACTOR = 0.2
elif scenario == 2:  # Winter peak Leading the way
    SCOTLAND_WIND_AVAILABILITY = 0.8
    NGET_WIND_AVAILABILITY = 0.8 * 0.7
    SCENARIO_NAME = 'Winter_{}_leading'.format(year)
    SOLAR_FACTOR = 0  # Winter evening
    CHP_FACTOR = 0.7


# Activate the necessary network variations for the considered year
variation_folder = app.GetProjectFolder("scheme")
Activevariations = app.GetActiveNetworkVariations()
for Activevariation in Activevariations:  # Start by deactivating all
    Activevariation.Deactivate()
for variation in variation_folder.GetContents():
    if year == 2031 :
        if variation.loc_name == '2030 NETWORK' or  variation.loc_name == 'SPf phase 2 (2030 network)' or  variation.loc_name == 'HVDC as 2-Terminal links & cable':
            variation.Activate()
    elif year == 2021:
        if variation.loc_name == 'SPf phase 2 (2021 network)':
            variation.Activate()

# Read network elements (note that they should be reloaded if a different network variation is activated) + sanity check
loads = app.GetCalcRelevantObjects("*.ElmLod")
loads = [load for load in loads if not load.outserv]
# loads = [load for load in loads if 'BESS' not in load.loc_name]
loads_incl_hvdc = loads
loads = [load for load in loads if load.loc_name != 'HVDC NET EMBED' and load.loc_name != 'HVDC WC load']  # Embedded HVDC link modelled as a load
loads = [load for load in loads if load.loc_name != 'HVDC NET IC']  # HVDC interconnections through England modelled as a load
dispatchable_loads = [load for load in loads if load.loc_name[:2] == 'H2' or load.loc_name[:4] == 'BESS']
loads = [load for load in loads if load.loc_name[:2] != 'H2' and load.loc_name[:4] != 'BESS']

scenario_data = {}
with open(os.path.join('..', 'FES data', 'aggregated', SCENARIO_NAME + '.csv')) as csv_file:
    reader = csv.reader(csv_file)
    next(reader)  # Skip header
    for row in reader:
        scenario_data[row[0]] = [float(value) for value in row[1:]]

for load in loads:
    data = scenario_data[load.loc_name[-4:]]
    storage = 0  # Included in gross load
    solar = data[3]
    wind = data[4]
    other = data[6]
    if load.cpZone.loc_name == 'NGET':
        wind_factor = NGET_WIND_AVAILABILITY
    else:
        wind_factor = SCOTLAND_WIND_AVAILABILITY
    load.plini = data[0] - solar * SOLAR_FACTOR - wind * wind_factor - other * CHP_FACTOR  # Netted DERs, #TODO: model explicitly
    load.qlini = data[1]  # Net Q
    if load.loc_name[-4:] == 'TUMM':
        load.qlini /= 2  # Necessary to get converging AC OPF, assumed to be handled by Tummel hydro power plant (not modelled)
    print(load.loc_name, load.plini, load.qlini)
print('Total load', sum([load.plini for load in loads]))

tfos = app.GetCalcRelevantObjects("*.ElmTr2")
tfos = [tfo for tfo in tfos if not tfo.outserv]
tfos = [tfo for tfo in tfos if tfo.IsEnergized()]
tfos = [tfo for tfo in tfos if tfo.bushv is not None and tfo.buslv is not None]
tfos = [tfo for tfo in tfos if tfo.bushv.IsClosed() and tfo.buslv.IsClosed()]

buses = app.GetCalcRelevantObjects("*.ElmTerm")
buses = [bus for bus in buses if not bus.outserv]
buses = [bus for bus in buses if bus.IsEnergized()]
buses = [bus for bus in buses if 'DC-BusBar' not in bus.loc_name]  # DC buses of statcoms
buses = [bus for bus in buses if 'T_HVDC_S' not in bus.loc_name]

lines = app.GetCalcRelevantObjects("*.ElmLne")
# lines = [line for line in lines if not line.outserv]
for line in lines:
    line.outserv = 0  # No lines are out of service in the base data, make sure they stay in service even if a previous call of the SCOPF was interupted
lines = [line for line in lines if line.IsEnergized()]
lines = [line for line in lines if line.bus1 is not None and line.bus2 is not None]
lines = [line for line in lines if line.bus1.IsClosed() and line.bus2.IsClosed()]

boundaries = app.GetCalcRelevantObjects('*.ElmBoundary')
if year == 2021:
    boundary_B4 = find_by_loc_name(boundaries, 'B4 Boundary')
elif year == 2030:
    boundary_B4 = find_by_loc_name(boundaries, 'B4 Boundary')
else:
    raise NotImplementedError('Year not considered')
boundary_B6 = find_by_loc_name(boundaries, 'B6 Boundary')

breakers = app.GetCalcRelevantObjects("*.ElmCoup")
breakers = [breaker for breaker in breakers if breaker.IsEnergized()]
breakers = [breaker for breaker in breakers if breaker.on_off]
breakers = [breaker for breaker in breakers if breaker.bus1 is not None and breaker.bus2 is not None]
breakers = [breaker for breaker in breakers if breaker.bus1.IsClosed() and breaker.bus2.IsClosed()]

series_capas = app.GetCalcRelevantObjects("*.ElmScap")
series_capas = [series_capa for series_capa in series_capas if not series_capa.outserv]
series_capas = [series_capa for series_capa in series_capas if series_capa.IsEnergized()]
series_capas = [series_capa for series_capa in series_capas if series_capa.bus1 is not None and series_capa.bus2 is not None]
series_capas = [series_capa for series_capa in series_capas if series_capa.bus1.IsClosed() and series_capa.bus2.IsClosed()]

# Buses
N_buses = len(buses)
bus_names = []
for bus in buses:
    bus_names.append(bus.loc_name)

# Branches
N_lines = len(lines)
N_tfos = len(tfos)
N_breakers = len(breakers)
N_series_capas = len(series_capas)
N_branches = N_lines + N_tfos + N_breakers + N_series_capas

branch_map = np.zeros((N_branches, N_buses))
B4_map = np.zeros((N_branches, len(boundary_B4.branches)))
B6_map = np.zeros((N_branches, len(boundary_B6.branches)))
# Branch data for DC OPF
admit = []
resist = []
# Branch data for AC OPF
Y = np.zeros((N_buses, N_buses), complex)
branch_FromFrom = np.zeros(N_branches, complex)
branch_FromTo = np.zeros(N_branches, complex)
branch_ToFrom = np.zeros(N_branches, complex)
branch_ToTo = np.zeros(N_branches, complex)


index = 0
B4_index = 0
B6_index = 0
for i in range(N_lines):
    bus_from = lines[i].bus1.GetParent().loc_name
    bus_to = lines[i].bus2.GetParent().loc_name
    branch_map[index, bus_names.index(bus_from)] = 1
    branch_map[index, bus_names.index(bus_to)] = -1

    if lines[i] in boundary_B4.branches:
        B4_map[index, B4_index] = 1
        B4_index += 1
    if lines[i] in boundary_B6.branches:
        B6_map[index, B6_index] = 1
        B6_index += 1

    X = max(lines[i].xSbasepu, 1e-6)  # Avoid division by 0
    admit.append(1 / X * lines[i].nlnum)
    resist.append(lines[i].rSbasepu / lines[i].nlnum)

    z = (lines[i].rSbasepu + 1j * lines[i].xSbasepu) / lines[i].nlnum
    if z == 0:
        z = 1j * 1e-6
    y1 = 1j * lines[i].bSbasepu * lines[i].nlnum / 2
    y2 = 1j * lines[i].bSbasepu * lines[i].nlnum / 2
    branch_FromFrom[index] = y1 + 1/z
    branch_FromTo[index] = -1/z
    branch_ToFrom[index] = -1/z
    branch_ToTo[index] = y2 + 1/z
    Y[bus_names.index(bus_from)][bus_names.index(bus_from)] += branch_FromFrom[index]
    Y[bus_names.index(bus_from)][bus_names.index(bus_to)] += branch_FromTo[index]
    Y[bus_names.index(bus_to)][bus_names.index(bus_from)] += branch_ToFrom[index]
    Y[bus_names.index(bus_to)][bus_names.index(bus_to)] += branch_ToTo[index]
    index += 1

for i in range(N_tfos):
    if tfos[i].bushv == None:
        print(tfos[i].loc_name)
    bus_from = tfos[i].bushv.GetParent().loc_name
    bus_to = tfos[i].buslv.GetParent().loc_name
    branch_map[index, bus_names.index(bus_from)] = 1
    branch_map[index, bus_names.index(bus_to)] = -1
    admit.append(1 / tfos[i].xSbasepu * tfos[i].ntnum)  # Sbase = tfo rating here
    resist.append(tfos[i].rSbasepu / tfos[i].ntnum)

    # Note that all transformers are modelled with tap fixed to 1pu and no magnetising impedance (lack of data)
    tfos[i].nntap = 0
    tfos[i].ntcrn = 0
    tfos[i].typ_id.nt2ag = 0

    z = (tfos[i].rSbasepu + 1j * tfos[i].xSbasepu) / tfos[i].ntnum
    y1 = 0
    y2 = 0
    branch_FromFrom[index] = y1 + 1/z
    branch_FromTo[index] = -1/z
    branch_ToFrom[index] = -1/z
    branch_ToTo[index] = y2 + 1/z
    Y[bus_names.index(bus_from)][bus_names.index(bus_from)] += branch_FromFrom[index]
    Y[bus_names.index(bus_from)][bus_names.index(bus_to)] += branch_FromTo[index]
    Y[bus_names.index(bus_to)][bus_names.index(bus_from)] += branch_ToFrom[index]
    Y[bus_names.index(bus_to)][bus_names.index(bus_to)] += branch_ToTo[index]
    index += 1

for i in range(N_breakers):
    bus_from = breakers[i].bus1.GetParent().loc_name
    bus_to = breakers[i].bus2.GetParent().loc_name
    try:
        branch_map[index, bus_names.index(bus_from)] = 1
    except ValueError as e:
        print(breakers[i].loc_name)
        raise e
    branch_map[index, bus_names.index(bus_to)] = -1
    admit.append(10000)
    resist.append(0)

    z = 1j / 10000
    branch_FromFrom[index] = 1/z
    branch_FromTo[index] = -1/z
    branch_ToFrom[index] = -1/z
    branch_ToTo[index] = 1/z
    Y[bus_names.index(bus_from)][bus_names.index(bus_from)] += branch_FromFrom[index]
    Y[bus_names.index(bus_from)][bus_names.index(bus_to)] += branch_FromTo[index]
    Y[bus_names.index(bus_to)][bus_names.index(bus_from)] += branch_ToFrom[index]
    Y[bus_names.index(bus_to)][bus_names.index(bus_to)] += branch_ToTo[index]
    index += 1

for i in range(N_series_capas):
    bus_from = series_capas[i].bus1.GetParent().loc_name
    bus_to = series_capas[i].bus2.GetParent().loc_name
    branch_map[index, bus_names.index(bus_from)] = 1
    branch_map[index, bus_names.index(bus_to)] = -1
    series_capas[i].ibypassed = 0
    Zb = series_capas[i].ucn**2 / baseMVA
    admit.append(-series_capas[i].bcap * Zb)
    resist.append(0)

    z = -1j * (1/series_capas[i].bcap) / Zb
    branch_FromFrom[index] = 1/z
    branch_FromTo[index] = -1/z
    branch_ToFrom[index] = -1/z
    branch_ToTo[index] = 1/z
    Y[bus_names.index(bus_from)][bus_names.index(bus_from)] += branch_FromFrom[index]
    Y[bus_names.index(bus_from)][bus_names.index(bus_to)] += branch_FromTo[index]
    Y[bus_names.index(bus_to)][bus_names.index(bus_from)] += branch_ToFrom[index]
    Y[bus_names.index(bus_to)][bus_names.index(bus_to)] += branch_ToTo[index]
    index += 1

G = Y.real
B = Y.imag
G_branch_FromFrom = branch_FromFrom.real
B_branch_FromFrom = branch_FromFrom.imag
G_branch_FromTo = branch_FromTo.real
B_branch_FromTo = branch_FromTo.imag
G_branch_ToFrom = branch_ToFrom.real
B_branch_ToFrom = branch_ToFrom.imag
G_branch_ToTo = branch_ToTo.real
B_branch_ToTo = branch_ToTo.imag
"""
for j in range(N_buses):
    if sum([abs(branch_map[i][j]) for i in range(N_branches)]) < 2:
        print('Bus', bus_names[j], 'is only connected to one branch, system thus cannot be N-1 secure')
        # raise Exception('Bus', bus_names[j], 'is only connected to one branch, system thus cannot be N-1 secure')
        # TODO: remove bus from buses and bus_names and branch_map
"""

branch_p_max = []
for line in lines:
    branch_p_max.append(line.typ_id.uline * line.Inom * 3**0.5 / baseMVA)  # Inom includes number of parallel lines (nlnum)
for tfo in tfos:
    branch_p_max.append(tfo.Snom / baseMVA)
for breaker in breakers:
    branch_p_max.append(999)
for series_capa in series_capas:
    branch_p_max.append(999)


# Loads
demand_bus = [0] * N_buses
demand_bus_Q = [0] * N_buses
for load in loads:
    load_bus = load.bus1.GetParent().loc_name
    demand_bus[bus_names.index(load_bus)] += load.plini / baseMVA
    demand_bus_Q[bus_names.index(load_bus)] += load.qlini / baseMVA

# Read generators (note that they should be reloaded if another network variation is activated)
sync_machines = app.GetCalcRelevantObjects("*.ElmSym")  # includes motor loads under ElmSym
for sync_machine in sync_machines:
    if sync_machine.loc_name == 'SG CCGT PEHE2':  # CCGT plant not dispatched in original data, allowed to be dispatched here if needed
        sync_machine.outserv = 0
sync_machines = [machine for machine in sync_machines if not machine.outserv]
sync_machines = [machine for machine in sync_machines if "Motor Load" not in machine.loc_name]
# sync_machines = [machine for machine in sync_machines if "SG LOI NGET4" not in machine.loc_name]  # Special machine to represent loss of infeed (i.e. event), not considered (outserv = 1)
sync_gens = [machine for machine in sync_machines if 'SYNC-COMP' not in machine.loc_name and 'SYNCON' not in machine.loc_name.upper()]
syncons = [machine for machine in sync_machines if 'SYNC-COMP' in machine.loc_name or 'SYNCON' in machine.loc_name.upper()]

"""
# Add outofstep variable to all synchronous machines for stability check
dyn_results = app.GetFromStudyCase('ElmRes').GetContents()
for machine in sync_machines:
    for result in dyn_results:
        if machine.loc_name == result.loc_name:
            if 's:outofstep' not in result.vars:
                result.vars += ['s:outofstep']  # Note: doesn't work, .append() doesn't work at all
"""

statcoms = app.GetCalcRelevantObjects("*.ElmVscmono")
statcoms = [statcom for statcom in statcoms if not statcom.outserv]

shunts = app.GetCalcRelevantObjects("*.ElmShnt")
shunts = [shunt for shunt in shunts if not shunt.outserv]
shunts = [shunt for shunt in shunts if 'DC-Cap' not in shunt.loc_name]

statvars = app.GetCalcRelevantObjects("*.ElmSvs")
statvars = [statvar for statvar in statvars if not statvar.outserv]

IBRs = app.GetCalcRelevantObjects("*.ElmGenstat")  # includes HVDCs under ElmGenstat
if year == 2030:
    find_by_loc_name(IBRs, 'DC BESS').pgini = 0 # 120 (*7) in original data
IBRs = [ibr for ibr in IBRs if ibr.loc_name != 'EFR BESS' and ibr.loc_name != 'DC BESS']
IBRs = [ibr for ibr in IBRs if not ibr.outserv]
wind_gens = [ibr for ibr in IBRs if 'HVDC' not in ibr.loc_name]
# HVDC_LCC = app.GetCalcRelevantObjects("*.ElmHvdclcc")  # Not used (outserv)
hvdc_links = [ibr for ibr in IBRs if 'HVDC' in ibr.loc_name]

hvdc_embedded_1 = []
hvdc_embedded_2 = []
hvdc_interconnections = []
hvdc_spit = []  # Wind farms connected via 2 hvdc links
for hvdc_link in hvdc_links:
    if hvdc_link.loc_name == "HVDC SPIT_BLHI2" or hvdc_link.loc_name == "HVDC SPIT_PEHE":
        hvdc_spit.append(hvdc_link)
    elif hvdc_link.loc_name[-2:] == '_A' or hvdc_link.loc_name[-3:] == '_1A' or hvdc_link.loc_name[-3:] == '_2A':
        hvdc_embedded_1.append(hvdc_link)
    elif hvdc_link.loc_name[-2:] == '_B' or hvdc_link.loc_name[-3:] == '_1B' or hvdc_link.loc_name[-3:] == '_2B':
        hvdc_embedded_2.append(hvdc_link)
    else:
        hvdc_interconnections.append(hvdc_link)
if len(hvdc_embedded_1) != len(hvdc_embedded_2):
    raise RuntimeError('Embedded HVDC links should have 2 ends each')
for hvdc_1, hvdc_2 in zip(hvdc_embedded_1, hvdc_embedded_2):
    if hvdc_1.loc_name[:-3] != hvdc_2.loc_name[:-3]:
        raise RuntimeError('Expected hvdc list to be sorted')

if len(hvdc_spit) != 2 and year == 2030:
    raise RuntimeError('Expected 2 hvdc links for SPIT')
elif len(hvdc_spit) != 1 and year == 2021:
    raise RuntimeError('Expected 1 hvdc link for SPIT')

N_sync_gens = len(sync_gens)
sync_gen_map = np.zeros((N_sync_gens, N_buses))
for i in range(N_sync_gens):
    sync_gen_map[i][bus_names.index(sync_gens[i].bus1.GetParent().loc_name)] = 1

N_wind_gens = len(wind_gens)
wind_gen_map = np.zeros((N_wind_gens, N_buses))
for i in range(N_wind_gens):
    wind_gen_map[i][bus_names.index(wind_gens[i].bus1.GetParent().loc_name)] = 1

N_syncons = len(syncons)
syncon_map = np.zeros((N_syncons, N_buses))
for i in range(N_syncons):
    syncon_map[i][bus_names.index(syncons[i].bus1.GetParent().loc_name)] = 1

N_statcoms = len(statcoms)
statcom_map = np.zeros((N_statcoms, N_buses))
for i in range(N_statcoms):
    statcom_map[i][bus_names.index(statcoms[i].busac.GetParent().loc_name)] = 1

N_shunts = len(shunts)
shunt_map = np.zeros((N_shunts, N_buses))
for i in range(N_shunts):
    shunt_map[i][bus_names.index(shunts[i].bus1.GetParent().loc_name)] = 1

N_statvars = len(statvars)
statvar_map = np.zeros((N_statvars, N_buses))
for i in range(N_statvars):
    statvar_map[i][bus_names.index(statvars[i].bus1.GetParent().loc_name)] = 1

N_hvdc_embedded = len(hvdc_embedded_1) + 1
hvdc_embedded_map = np.zeros((N_hvdc_embedded, N_buses))
i = -1  # No embedded hvdc links in 2021 scenario
for i in range(N_hvdc_embedded - 1):
    hvdc_embedded_map[i][bus_names.index(hvdc_embedded_1[i].bus1.GetParent().loc_name)] = 1
    hvdc_embedded_map[i][bus_names.index(hvdc_embedded_2[i].bus1.GetParent().loc_name)] = -1  # Neglect losses (depend on flow direction + small and most of them occur at NGET bus)
hvdc_embedded_map[i+1][bus_names.index(find_by_loc_name(loads_incl_hvdc, 'HVDC WC load').bus1.GetParent().loc_name)] = 1
hvdc_embedded_map[i+1][bus_names.index(find_by_loc_name(loads_incl_hvdc, 'HVDC NET EMBED').bus1.GetParent().loc_name)] = -1

N_hvdc_embedded_Q = len(hvdc_embedded_1)  # Separate maps for reactive power as it can be controlled independently at both side of the links (VSC converters) + does not consider Western interconnector (LCC)
hvdc_embedded_map_Q1 = np.zeros((N_hvdc_embedded_Q, N_buses))
for i in range(N_hvdc_embedded_Q):
    hvdc_embedded_map_Q1[i][bus_names.index(hvdc_embedded_1[i].bus1.GetParent().loc_name)] = 1
hvdc_embedded_map_Q2 = np.zeros((N_hvdc_embedded_Q, N_buses))
for i in range(N_hvdc_embedded_Q):
    hvdc_embedded_map_Q2[i][bus_names.index(hvdc_embedded_2[i].bus1.GetParent().loc_name)] = 1

N_hvdc_interconnections = len(hvdc_interconnections) + 1
hvdc_interconnection_map = np.zeros((N_hvdc_interconnections, N_buses))
for i in range(N_hvdc_interconnections - 1):
    hvdc_interconnection_map[i][bus_names.index(hvdc_interconnections[i].bus1.GetParent().loc_name)] = 1
hvdc_interconnection_map[i+1][bus_names.index('NGET4')] = 1

N_hvdc_spit = len(hvdc_spit)
hvdc_spit_map = np.zeros((N_hvdc_spit, N_buses))
for i in range(N_hvdc_spit):
    hvdc_spit_map[i][bus_names.index(hvdc_spit[i].bus1.GetParent().loc_name)] = 1

N_dispatchable_loads = len(dispatchable_loads)
dispatchable_load_map = np.zeros((N_dispatchable_loads, N_buses))
for i in range(N_dispatchable_loads):
    dispatchable_load_map[i][bus_names.index(dispatchable_loads[i].bus1.GetParent().loc_name)] = 1

sync_min = []
sync_max = []
sync_Qmin = []
sync_Qmax = []
droop = []
sync_gen_costs = []
for sync_gen in sync_gens:
    # Pmin
    if sync_gen.loc_name == 'SG HC SLACK NGET4':
        sync_min.append(0)
    elif sync_gen.cCategory == 'Hydro' or sync_gen.cCategory == 'Others':  # SG PSH WIYH2
        if 'PSH' in sync_gen.loc_name:
            sync_min.append(-sync_gen.P_max * sync_gen.ngnum / baseMVA)
        else:
            sync_min.append(0)
    elif sync_gen.cCategory == 'Nuclear':
        sync_min.append(sync_gen.P_max * sync_gen.ngnum / baseMVA * 0.9)  # Missing data for Pmin, so use 40% (90% for nuclear)
    else:
        sync_min.append(sync_gen.P_max * sync_gen.ngnum / baseMVA * 0.4)

    if sync_gen.loc_name == 'SG HC SLACK NGET4':  # Slack generator for mismatch between OPF and Powerfactory, should not be used directly in OPF
        sync_min[-1] = 0  # Minimum from operational limits and actual value from original data
        sync_max.append(0 / baseMVA)
        sync_Qmin.append(0)
        sync_Qmax.append(0)
        droop.append(1)
    elif sync_gen.loc_name == 'SG FR NGET4':  # frequency response plant
        sync_min[-1] = 100 / baseMVA  # Minimum from operational limits and actual value from original data
        sync_max.append(100 / baseMVA)
        sync_Qmin.append(0)
        sync_Qmax.append(0)
        droop.append(0)
    else:
        sync_max.append(sync_gen.P_max * sync_gen.ngnum / baseMVA)
        sync_Qmin.append(-sync_gen.P_max * sync_gen.ngnum / baseMVA * (1-0.9**2)**0.5)  # Missing data for Q, so assume a 0.9 pf
        sync_Qmax.append(sync_gen.P_max * sync_gen.ngnum / baseMVA * (1-0.9**2)**0.5)
        droop.append(0)

    if sync_gen.cCategory == 'Hydro' or sync_gen.cCategory == 'Others':  # SG PSH WIYH2
        sync_gen_costs.append(20 * baseMVA)
    elif sync_gen.cCategory == 'Gas':
        sync_gen_costs.append(80 * baseMVA)
    elif sync_gen.cCategory == 'Nuclear':
        sync_gen_costs.append(-100 * baseMVA)  # Difficult to constaint-off
    else:
        raise NotImplementedError(sync_gen.cCategory)

hvdc_interconnection_costs = []
for hvdc in hvdc_interconnections:
    hvdc_interconnection_costs.append(50 * baseMVA)
# Sum of HVDC connections via England (modelled as load)
hvdc_interconnection_costs.append(50 * baseMVA)

dispatchable_load_costs = []
for load in dispatchable_loads:
    dispatchable_load_costs.append(40 * baseMVA)


wind_max = []
wind_Qmin = []
wind_Qmax = []
for wind_gen in wind_gens:
    if wind_gen.cpZone.loc_name == 'NGET':
        wind_max.append(wind_gen.sgn * wind_gen.cosn * wind_gen.ngnum / baseMVA * NGET_WIND_AVAILABILITY)
    else:
        wind_max.append(wind_gen.sgn * wind_gen.cosn * wind_gen.ngnum / baseMVA * SCOTLAND_WIND_AVAILABILITY)
    wind_Qmin.append(-wind_gen.sgn * (1-wind_gen.cosn**2)**0.5 * wind_gen.ngnum / baseMVA)
    wind_Qmax.append(wind_gen.sgn * (1-wind_gen.cosn**2)**0.5 * wind_gen.ngnum / baseMVA)

syncon_Qmin = []
syncon_Qmax = []
for syncon in syncons:
    syncon_Qmin.append(syncon.cQ_min * syncon.ngnum / baseMVA)
    syncon_Qmax.append(syncon.cQ_max * syncon.ngnum / baseMVA)

statcom_Qmin = []
statcom_Qmax = []
for statcom in statcoms:
    statcom_Qmin.append(statcom.cQ_min * statcom.nparnum / baseMVA)
    statcom_Qmax.append(statcom.cQ_max * statcom.nparnum / baseMVA)

shunt_Qmin = []
shunt_Qmax = []
shunt_abs_max = []
for shunt in shunts:  # Assume continuous control of shunts is possible for numerical stability of the OPF. It can be noted that most shunts are in parallel with a statcom or a syncon. TODO: redo OPF with shunts fixed to nearest integer value?
    if shunt.ncapx < 100:
        shunt.ncapx *= 100
        if shunt.qrean is not None:
            shunt.qrean /= 100
        if shunt.qcapn is not None:
            shunt.qcapn /= 100
    shunt.iswitch = False
    shunt_abs_max.append(shunt.Qmax / baseMVA)
    if shunt.shtype == 1:  # Inductance (R-L)
        shunt_Qmin.append(-shunt.Qmax / baseMVA)
        shunt_Qmax.append(0)
    elif shunt.shtype == 2:  # Capacitance (C)
        shunt_Qmin.append(0)
        shunt_Qmax.append(shunt.Qmax / baseMVA)
    else:
        raise NotImplementedError('Only consider R-L and C shunts')

statvar_Qmin = []
statvar_Qmax = []
for statvar in statvars:
    statvar_Qmin.append(statvar.qmin / baseMVA)
    if statvar.loc_name == 'SVC HUNE4':
        statvar_Qmax.append(-statvar.qmin / baseMVA)  # Has a 2 GVaR inductive capability for some reason, so use -Qmin instead
    else:
        statvar_Qmax.append(statvar.qmax / baseMVA)

hvdc_embedded_min = []
hvdc_embedded_max = []
for hvdc_embedded in hvdc_embedded_1:
    hvdc_max = hvdc_embedded.sgn * hvdc_embedded.cosn * hvdc_embedded.ngnum / baseMVA
    hvdc_embedded_min.append(-hvdc_max)
    hvdc_embedded_max.append(hvdc_max)
# Western interconnector (modelled as load without reactive power)
hvdc_embedded_min.append(-2000/baseMVA)
hvdc_embedded_max.append(2000/baseMVA)

hvdc_embedded_Qmin = []
hvdc_embedded_Qmax = []
for hvdc_embedded in hvdc_embedded_1:  # Assume same converter rating on both side of links
    hvdc_embedded_Qmin.append(- hvdc_embedded.sgn * (1-hvdc_embedded.cosn**2)**0.5 * hvdc_embedded.ngnum / baseMVA)
    hvdc_embedded_Qmax.append(hvdc_embedded.sgn * (1-hvdc_embedded.cosn**2)**0.5 * hvdc_embedded.ngnum / baseMVA)


hvdc_interconnection_min = []
hvdc_interconnection_max = []
hvdc_interconnection_Qmin = []
hvdc_interconnection_Qmax = []
for hvdc_interconnection in hvdc_interconnections:
    hvdc_max = hvdc_interconnection.sgn * hvdc_interconnection.cosn * hvdc_interconnection.ngnum / baseMVA
    if hvdc_interconnection.loc_name == 'HVDC IC NSL':
        hvdc_max = 1400 / baseMVA
    hvdc_interconnection_min.append(-hvdc_max)
    hvdc_interconnection_max.append(hvdc_max)
    hvdc_interconnection_Qmin.append(- hvdc_interconnection.sgn * (1-hvdc_interconnection.cosn**2)**0.5 * hvdc_interconnection.ngnum / baseMVA)
    hvdc_interconnection_Qmax.append(hvdc_interconnection.sgn * (1-hvdc_interconnection.cosn**2)**0.5 * hvdc_interconnection.ngnum / baseMVA)
# Sum of HVDC connections via England (modelled as load)
if year == 2021:
    hvdc_max = 6000 / baseMVA
elif year == 2030:
    hvdc_max = (15900 - 2800) / baseMVA
else:
    raise NotImplementedError('Year not modelled')
hvdc_interconnection_min.append(-hvdc_max)
hvdc_interconnection_max.append(hvdc_max)
hvdc_interconnection_Qmin.append(0)
hvdc_interconnection_Qmax.append(0)

hvdc_spit_min = []
hvdc_spit_max = []
hvdc_spit_Qmin = []
hvdc_spit_Qmax = []
spit_total_max = 2500 / baseMVA * SCOTLAND_WIND_AVAILABILITY
for hvdc in hvdc_spit:
    hvdc_max = hvdc.sgn * hvdc.cosn * hvdc.ngnum / baseMVA
    hvdc_spit_min.append(0.2 * spit_total_max)  # Export at least part of the power through both HVDCs
    hvdc_spit_max.append(hvdc_max)
    hvdc_spit_Qmin.append(- hvdc.sgn * (1-hvdc.cosn**2)**0.5 * hvdc.ngnum / baseMVA)
    hvdc_spit_Qmax.append(hvdc.sgn * (1-hvdc.cosn**2)**0.5 * hvdc.ngnum / baseMVA)

csv_path = 'dispatchable_loads_max.csv'
"""
if not os.path.isfile(csv_path):
    print('Saving current load dispatch as maximum demand for dispatchable loads (i.e. BESS and H2)')  # 5GW at NGET, and 5GW in Scotland (5 * 1/7 + 2/7 at HUNE4), assume uniform for BESS, total 4.8 GW
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for load in dispatchable_loads:
            P = load.plini
            writer.writerow([load.loc_name, P])
"""
with open(csv_path, newline='') as csv_file:
    reader = csv.reader(csv_file)
    csv_content = {}
    for row in reader:
        csv_content[row[0]] = row[1]

dispatchable_load_min = []
dispatchable_load_max = []
for load in dispatchable_loads:
    P = float(csv_content[load.loc_name])
    if 'BESS' in load.loc_name:
        dispatchable_load_min.append(-P / baseMVA)
    else:
        dispatchable_load_min.append(0)
    dispatchable_load_max.append(P / baseMVA)


# DCOPF: Send data to GAMS
losses = 0.04  # Loss estimation used in DC OPF to minimise differences between AC and DC formulations
print('\nPSCDCOPF')
dcopf_path = os.path.join('a-PSCDCOPF') # , str(hour))
Path(dcopf_path).mkdir(parents=True, exist_ok=True)
ws = gams.GamsWorkspace(working_directory=os.path.join(os.getcwd(), dcopf_path), debug=gams.DebugLevel.Off)
db_preDC = ws.add_database()

def addGamsSet(db, name, description, lst):
    # Adds a 1-dimensional set
    set = db.add_set(name, 1, description)
    for i in lst:
        set.add_record(str(i))
    return set

# N_lines = 1  # Removes (almost all) N-1 constraints (test only)
i_sync = addGamsSet(db_preDC, 'i_sync', 'sync generators', range(1, N_sync_gens + 1))
i_wind = addGamsSet(db_preDC, 'i_wind', 'wind generators', range(1, N_wind_gens + 1))
i_syncon = addGamsSet(db_preDC, 'i_syncon', 'syncon generators', range(1, N_syncons + 1))
i_bus = addGamsSet(db_preDC, 'i_bus', 'buses', range(1, N_buses + 1))
i_branch = addGamsSet(db_preDC, 'i_branch', 'branches', range(1, N_branches + 1))
i_contingency = addGamsSet(db_preDC, 'i_contingency', 'lines', range(1, N_lines + 1))
i_hvdc_embedded = addGamsSet(db_preDC, 'i_hvdc_embedded', 'embedded hvdc links', range(1, N_hvdc_embedded + 1))
i_hvdc_interconnection = addGamsSet(db_preDC, 'i_hvdc_interconnection', 'hvdc interconnections', range(1, N_hvdc_interconnections + 1))
i_hvdc_spit = addGamsSet(db_preDC, 'i_hvdc_spit', 'hvdc spits', range(1, N_hvdc_spit + 1))
i_dispatchable_load = addGamsSet(db_preDC, 'i_dispatchable_load', 'dispatchable loads', range(1, N_dispatchable_loads + 1))

def addGamsParams(db, name, description, sets, values):
    m = db.add_parameter_dc(name, sets, description)
    if len(sets) == 1:
        i_1 = sets[0]
        for i in range(len(i_1)):
            m.add_record(str(i+1)).value = values[i]
    elif len(sets) == 2:
        i_1, i_2 = sets[0], sets[1]
        for i in range(len(i_1)):
            for j in range(len(i_2)):
                m.add_record((str(i+1),str(j+1))).value = values[i][j]

addGamsParams(db_preDC, 'sync_map', 'sync generators map', [i_sync, i_bus], sync_gen_map)
addGamsParams(db_preDC, 'wind_map', 'wind generators map', [i_wind, i_bus], wind_gen_map)
# addGamsParams(db_preDC, 'syncon_map', 'syncon generators map', [i_syncon, i_bus], syncon_gen_map)  # Syncons not considered in the DC approximation
addGamsParams(db_preDC, 'branch_map', 'branches map', [i_branch, i_bus], branch_map)
addGamsParams(db_preDC, 'hvdc_embedded_map', 'hvdc_embedded map', [i_hvdc_embedded, i_bus], hvdc_embedded_map)
addGamsParams(db_preDC, 'hvdc_interconnection_map', 'hvdc_interconnection map', [i_hvdc_interconnection, i_bus], hvdc_interconnection_map)
addGamsParams(db_preDC, 'hvdc_spit_map', 'hvdc_spit map', [i_hvdc_spit, i_bus], hvdc_spit_map)
addGamsParams(db_preDC, 'dispatchable_load_map', 'dispatchable_load map', [i_dispatchable_load, i_bus], dispatchable_load_map)

addGamsParams(db_preDC, 'sync_min', 'sync generator minimum generation', [i_sync], sync_min)
addGamsParams(db_preDC, 'sync_max', 'sync generator maximum generation', [i_sync], sync_max)
addGamsParams(db_preDC, 'wind_max', 'wind generator maximum generation', [i_wind], wind_max)
addGamsParams(db_preDC, 'hvdc_embedded_min', 'embedded hvdc minimum generation', [i_hvdc_embedded], hvdc_embedded_min)
addGamsParams(db_preDC, 'hvdc_embedded_max', 'embedded hvdc maximum generation', [i_hvdc_embedded], hvdc_embedded_max)
addGamsParams(db_preDC, 'hvdc_interconnection_min', 'interconnection hvdc minimum generation', [i_hvdc_interconnection], hvdc_interconnection_min)
addGamsParams(db_preDC, 'hvdc_interconnection_max', 'interconnection hvdc maximum generation', [i_hvdc_interconnection], hvdc_interconnection_max)
addGamsParams(db_preDC, 'hvdc_spit_min', 'spit hvdc minimum generation', [i_hvdc_spit], hvdc_spit_min)
addGamsParams(db_preDC, 'hvdc_spit_max', 'spit hvdc maximum generation', [i_hvdc_spit], hvdc_spit_max)
db_preDC.add_parameter("hvdc_spit_total_max", 0, 'available wind for spit hvdcs').add_record().value = spit_total_max
addGamsParams(db_preDC, 'dispatchable_load_min', 'dispatchable load minimum generation', [i_dispatchable_load], dispatchable_load_min)
addGamsParams(db_preDC, 'dispatchable_load_max', 'dispatchable load maximum generation', [i_dispatchable_load], dispatchable_load_max)

addGamsParams(db_preDC, 'branch_admittance', 'branch admittance', [i_branch], admit)
addGamsParams(db_preDC, 'branch_resistance', 'branch resistance', [i_branch], resist)
addGamsParams(db_preDC, 'branch_max_N', 'Normal branch max power', [i_branch], np.array(branch_p_max))
addGamsParams(db_preDC, 'branch_max_E', 'Emergency branch max power', [i_branch], np.array(branch_p_max))

contingency_states = np.ones((N_branches, N_branches)) - np.diag(np.diag(np.ones((N_branches, N_branches))))  # Matrix full of ones, but zeroes on the diagonal
contingency_states = contingency_states[:, :N_lines]  # Only consider failure of lines, not transformers, series capa, etc.
addGamsParams(db_preDC, 'contingency_states', 'Line states in the considered contingencies', [i_branch, i_contingency], contingency_states)

addGamsParams(db_preDC, 'demand', 'demand at each bus', [i_bus], np.array(demand_bus) * (1 + losses))

addGamsParams(db_preDC, 'lincost', 'linear cost', [i_sync], sync_gen_costs)
addGamsParams(db_preDC, 'hvdc_interconnection_costs', 'import cost', [i_hvdc_interconnection], hvdc_interconnection_costs)
addGamsParams(db_preDC, 'dispatchable_load_costs', 'dispatchable load costs', [i_dispatchable_load], dispatchable_load_costs)

db_preDC.export('PrePSCDCOPF.gdx')
t = ws.add_job_from_file('PSCDCOPF.gms')
t.run()

db_postDC = ws.add_database_from_gdx("PostPSCDCOPF.gdx")

solve_status = int(db_postDC["sol"].first_record().value)
if solve_status != 1 and solve_status != 2 and solve_status != 7 and solve_status != 8 and solve_status != 6:
    raise RuntimeError('PSCDCOPF: no solution found, error code:', solve_status)

P_DC_sync = list({rec.keys[0]:rec.level for rec in db_postDC["P_sync"]}.values())
P_DC_wind = list({rec.keys[0]:rec.level for rec in db_postDC["P_wind"]}.values())
P_DC_hvdc_embedded = list({rec.keys[0]:rec.level for rec in db_postDC["P_hvdc_embedded"]}.values())
P_DC_hvdc_interconnection = list({rec.keys[0]:rec.level for rec in db_postDC["P_hvdc_interconnection"]}.values())
P_DC_hvdc_spit = list({rec.keys[0]:rec.level for rec in db_postDC["P_hvdc_spit"]}.values())
P_DC_dispatchable_load = list({rec.keys[0]:rec.level for rec in db_postDC["P_dispatchable_load"]}.values())
P_DC_pf = list({rec.keys[0]:rec.level for rec in db_postDC["pf0"]}.values())
on_DC = {rec.keys[0]:rec.level for rec in db_postDC["on"]}
for i in range(1, N_sync_gens + 1):
    if on_DC.get('{}'.format(i)) is None:
        on_DC['{}'.format(i)] = 0  # Readd values that GAMS deletes for some reason
on_DC = list(on_DC.values())
for i, sync_gen in enumerate(sync_gens):
    if sync_gen.cCategory == 'Hydro' or sync_gen.cCategory == 'Others':  # SG PSH WIYH2
        on_DC[i] = 1  # Do not disconnect hydro plants

pf_DC = {rec.keys[0]:rec.level for rec in db_postDC["pf0"]}
theta_DC = list({rec.keys[0]:rec.level for rec in db_postDC["theta0"]}.values())

cost = db_postDC["total_cost"].first_record().level

print('Total cost:', round(cost, 2))
print('Sync gen:', sum(P_DC_sync) * baseMVA / 1000, ' / ', sum(sync_max) * baseMVA / 1000, 'GW')
print('Wind gen:', (sum(P_DC_wind) + sum(P_DC_hvdc_spit)) * baseMVA / 1000, ' / ', (sum(wind_max) + spit_total_max) * baseMVA / 1000, 'GW')
print('Direct imports:', sum(list(P_DC_hvdc_interconnection)[:-1]) * baseMVA / 1000, ' / ', sum(hvdc_interconnection_max[:-1]) * baseMVA / 1000, 'GW (negative if exporting)')
print('Imports via England:', sum(list(P_DC_hvdc_interconnection)[-1:]) * baseMVA / 1000, ' / ', sum(hvdc_interconnection_max[-1:]) * baseMVA / 1000, 'GW (negative if exporting)')

total_generation = sum(P_DC_sync) + sum(P_DC_wind) + sum(P_DC_hvdc_spit)
total_demand = sum(demand_bus) * (1 + losses) - sum(P_DC_hvdc_interconnection) + sum(P_DC_dispatchable_load)
print('Total generation:', total_generation * baseMVA / 1000, 'GW')
print('Price-responsive load:', sum(P_DC_dispatchable_load) * baseMVA / 1000, ' / ', sum(dispatchable_load_max) * baseMVA / 1000, 'GW')
print('Generation/load imbalance (should be 0 in DC approx):', (total_generation - total_demand) * baseMVA / 1000, 'GW')
print('Total embedded HVDC flows (North-South):', -sum(P_DC_hvdc_embedded) * baseMVA / 1000, ' / ', sum(hvdc_embedded_max) * baseMVA / 1000, 'GW')

# print((abs(pf_DC['24']) + abs(pf_DC['25']) + abs(pf_DC['26']) + abs(pf_DC['27']) +  abs(pf_DC['30'])*2)*baseMVA/1000)  # AC flows parallel to HVDC embedded (through B7 boundary)


#####
# ACOPF
#####
print('\nACOPF')

sync_min = sync_min * np.array(on_DC)  # Keep commited status of sync gens from DC OPF (avoid mixed integer with AC formulation)
sync_max = sync_max * np.array(on_DC)
sync_Qmin = sync_Qmin * np.array(on_DC)
sync_Qmax = sync_Qmax * np.array(on_DC)

acopf_path = os.path.join('b-ACOPF')  #, str(hour))
Path(acopf_path).mkdir(parents=True, exist_ok=True)
ws = gams.GamsWorkspace(working_directory=os.path.join(os.getcwd(), acopf_path), debug=gams.DebugLevel.Off)
db_preAC = ws.add_database()

i_sync = addGamsSet(db_preAC, 'i_sync', 'sync generators', range(1, N_sync_gens + 1))
i_wind = addGamsSet(db_preAC, 'i_wind', 'wind generators', range(1, N_wind_gens + 1))
i_syncon = addGamsSet(db_preAC, 'i_syncon', 'syncon generators', range(1, N_syncons + 1))
i_statcom = addGamsSet(db_preAC, 'i_statcom', 'statcoms', range(1, N_statcoms + 1))
i_shunt = addGamsSet(db_preAC, 'i_shunt', 'shunts', range(1, N_shunts + 1))
i_statvar = addGamsSet(db_preAC, 'i_statvar', 'statvars', range(1, N_statvars + 1))
i_bus = addGamsSet(db_preAC, 'i_bus', 'buses', range(1, N_buses + 1))
i_branch = addGamsSet(db_preAC, 'i_branch', 'branches', range(1, N_branches + 1))
# i_contingency = addGamsSet(db_preAC, 'i_contingency', 'lines', range(1, N_lines + 1))
i_hvdc_embedded = addGamsSet(db_preAC, 'i_hvdc_embedded', 'embedded hvdc links', range(1, N_hvdc_embedded + 1))
i_hvdc_embedded_Q = addGamsSet(db_preAC, 'i_hvdc_embedded_Q', 'embedded hvdc links reactive control', range(1, N_hvdc_embedded_Q + 1))
i_hvdc_interconnection = addGamsSet(db_preAC, 'i_hvdc_interconnection', 'hvdc interconnections', range(1, N_hvdc_interconnections + 1))
i_hvdc_spit = addGamsSet(db_preAC, 'i_hvdc_spit', 'hvdc spits', range(1, N_hvdc_spit + 1))
i_dispatchable_load = addGamsSet(db_preAC, 'i_dispatchable_load', 'hvdc spits', range(1, N_dispatchable_loads + 1))

addGamsParams(db_preAC, 'sync_map', 'sync generators map', [i_sync, i_bus], sync_gen_map)
addGamsParams(db_preAC, 'wind_map', 'wind generators map', [i_wind, i_bus], wind_gen_map)
addGamsParams(db_preAC, 'syncon_map', 'syncon generators map', [i_syncon, i_bus], syncon_map)
addGamsParams(db_preAC, 'statcom_map', 'statcom generators map', [i_statcom, i_bus], statcom_map)
addGamsParams(db_preAC, 'shunt_map', 'shunt generators map', [i_shunt, i_bus], shunt_map)
addGamsParams(db_preAC, 'statvar_map', 'statvar generators map', [i_statvar, i_bus], statvar_map)
addGamsParams(db_preAC, 'branch_map', 'branches map', [i_branch, i_bus], branch_map)
addGamsParams(db_preAC, 'hvdc_embedded_map', 'hvdc_embedded map', [i_hvdc_embedded, i_bus], hvdc_embedded_map)
addGamsParams(db_preAC, 'hvdc_embedded_map_Q1', 'hvdc_embedded_Q1 map', [i_hvdc_embedded_Q, i_bus], hvdc_embedded_map_Q1)
addGamsParams(db_preAC, 'hvdc_embedded_map_Q2', 'hvdc_embedded_Q2 map', [i_hvdc_embedded_Q, i_bus], hvdc_embedded_map_Q2)
addGamsParams(db_preAC, 'hvdc_interconnection_map', 'hvdc_interconnection map', [i_hvdc_interconnection, i_bus], hvdc_interconnection_map)
addGamsParams(db_preAC, 'hvdc_spit_map', 'hvdc_spit map', [i_hvdc_spit, i_bus], hvdc_spit_map)
addGamsParams(db_preAC, 'dispatchable_load_map', 'dispatchable_load map', [i_dispatchable_load, i_bus], dispatchable_load_map)

addGamsParams(db_preAC, 'sync_min', 'sync generator minimum generation', [i_sync], sync_min)
addGamsParams(db_preAC, 'sync_max', 'sync generator maximum generation', [i_sync], sync_max)
addGamsParams(db_preAC, 'sync_Qmin', 'sync generator minimum reactive generation', [i_sync], sync_Qmin)
addGamsParams(db_preAC, 'sync_Qmax', 'sync generator maximum reactive generation', [i_sync], sync_Qmax)

addGamsParams(db_preAC, 'wind_max', 'wind generator maximum generation', [i_wind], wind_max)
addGamsParams(db_preAC, 'wind_Qmin', 'wind generator minimum reactive power', [i_wind], wind_Qmin)
addGamsParams(db_preAC, 'wind_Qmax', 'wind generator maximum reactive power', [i_wind], wind_Qmax)

addGamsParams(db_preAC, 'syncon_Qmin', 'syncon minimum reactive generation', [i_syncon], syncon_Qmin)
addGamsParams(db_preAC, 'syncon_Qmax', 'syncon maximum reactive generation', [i_syncon], syncon_Qmax)

addGamsParams(db_preAC, 'statcom_Qmin', 'statcom minimum reactive generation', [i_statcom], statcom_Qmin)
addGamsParams(db_preAC, 'statcom_Qmax', 'statcom maximum reactive generation', [i_statcom], statcom_Qmax)

addGamsParams(db_preAC, 'shunt_Qmin', 'shunt minimum reactive generation', [i_shunt], shunt_Qmin)
addGamsParams(db_preAC, 'shunt_Qmax', 'shunt maximum reactive generation', [i_shunt], shunt_Qmax)

addGamsParams(db_preAC, 'statvar_Qmin', 'statvar minimum reactive generation', [i_statvar], statvar_Qmin)
addGamsParams(db_preAC, 'statvar_Qmax', 'statvar maximum reactive generation', [i_statvar], statvar_Qmax)

addGamsParams(db_preAC, 'hvdc_embedded_min', 'embedded hvdc minimum generation', [i_hvdc_embedded], hvdc_embedded_min)
addGamsParams(db_preAC, 'hvdc_embedded_max', 'embedded hvdc maximum generation', [i_hvdc_embedded], hvdc_embedded_max)
addGamsParams(db_preAC, 'hvdc_embedded_Qmin', 'embedded hvdc minimum reactive generation', [i_hvdc_embedded_Q], hvdc_embedded_Qmin)
addGamsParams(db_preAC, 'hvdc_embedded_Qmax', 'embedded hvdc maximum reactive generation', [i_hvdc_embedded_Q], hvdc_embedded_Qmax)
addGamsParams(db_preAC, 'hvdc_interconnection_min', 'interconnection hvdc minimum generation', [i_hvdc_interconnection], hvdc_interconnection_min)
addGamsParams(db_preAC, 'hvdc_interconnection_max', 'interconnection hvdc maximum generation', [i_hvdc_interconnection], hvdc_interconnection_max)
addGamsParams(db_preAC, 'hvdc_interconnection_Qmin', 'interconnection hvdc minimum reactive generation', [i_hvdc_interconnection], hvdc_interconnection_Qmin)
addGamsParams(db_preAC, 'hvdc_interconnection_Qmax', 'interconnection hvdc maximum reactive generation', [i_hvdc_interconnection], hvdc_interconnection_Qmax)
addGamsParams(db_preAC, 'hvdc_spit_min', 'spit hvdc minimum generation', [i_hvdc_spit], hvdc_spit_min)
addGamsParams(db_preAC, 'hvdc_spit_max', 'spit hvdc maximum generation', [i_hvdc_spit], hvdc_spit_max)
db_preAC.add_parameter("hvdc_spit_total_max", 0, 'available wind for spit hvdcs').add_record().value = spit_total_max
addGamsParams(db_preAC, 'hvdc_spit_Qmin', 'spit hvdc minimum reactive generation', [i_hvdc_spit], hvdc_spit_Qmin)
addGamsParams(db_preAC, 'hvdc_spit_Qmax', 'spit hvdc maximum reactive generation', [i_hvdc_spit], hvdc_spit_Qmax)
addGamsParams(db_preAC, 'dispatchable_load_min', 'dispatchable load minimum generation', [i_dispatchable_load], dispatchable_load_min)
addGamsParams(db_preAC, 'dispatchable_load_max', 'dispatchable load maximum generation', [i_dispatchable_load], dispatchable_load_max)

addGamsParams(db_preAC, 'G', 'conductance matrix', [i_bus, i_bus], G)
addGamsParams(db_preAC, 'B', 'susceptance matrix', [i_bus, i_bus], B)
addGamsParams(db_preAC, 'Gff', 'line conductance (from-from)', [i_branch], G_branch_FromFrom)
addGamsParams(db_preAC, 'Gft', 'line conductance (from-to)', [i_branch], G_branch_FromTo)
addGamsParams(db_preAC, 'Bff', 'line susceptance (from-from)', [i_branch], B_branch_FromFrom)
addGamsParams(db_preAC, 'Bft', 'line susceptance (from-to)', [i_branch], B_branch_FromTo)
addGamsParams(db_preAC, 'branch_max_N', 'Normal branch max power', [i_branch], np.array(branch_p_max))

addGamsParams(db_preAC, 'demand', 'demand at each bus', [i_bus], demand_bus)
addGamsParams(db_preAC, 'demandQ', 'reactive demand at each bus', [i_bus], demand_bus_Q)

addGamsParams(db_preAC, 'lincost', 'linear cost', [i_sync], sync_gen_costs)
addGamsParams(db_preAC, 'hvdc_interconnection_costs', 'import cost', [i_hvdc_interconnection], hvdc_interconnection_costs)
addGamsParams(db_preAC, 'dispatchable_load_costs', 'dispatchable load costs', [i_dispatchable_load], dispatchable_load_costs)

addGamsParams(db_preAC, 'P_sync_0', 'Initial sync outputs', [i_sync], list(P_DC_sync))
addGamsParams(db_preAC, 'P_wind_0', 'Initial wind outputs', [i_wind], list(P_DC_wind))
addGamsParams(db_preAC, 'P_hvdc_embedded_0', 'Initial embedded hvdc flows', [i_hvdc_embedded], list(P_DC_hvdc_embedded))
addGamsParams(db_preAC, 'P_hvdc_interconnection_0', 'Initial hvdc interconection flows', [i_hvdc_interconnection], list(P_DC_hvdc_interconnection))
addGamsParams(db_preAC, 'P_hvdc_spit_0', 'Initial hvdc interconection flows', [i_hvdc_spit], list(P_DC_hvdc_spit))
addGamsParams(db_preAC, 'P_dispatchable_load_0', 'Initial hvdc interconection flows', [i_dispatchable_load], list(P_DC_dispatchable_load))
addGamsParams(db_preAC, 'Ppf_0', 'Initial line active power flows', [i_branch], list(P_DC_pf))

db_preAC.export('PreACOPF.gdx')
t = ws.add_job_from_file('ACOPF.gms')
t.run()

db_postAC = ws.add_database_from_gdx("PostACOPF.gdx")

solve_status = int(db_postAC["sol"].first_record().value)
if solve_status != 1 and solve_status != 2 and solve_status != 7:
    raise RuntimeError('ACOPF: no solution found, error code:', solve_status)

P_AC_sync = list({rec.keys[0]:rec.level for rec in db_postAC["P_sync"]}.values())
P_AC_wind = list({rec.keys[0]:rec.level for rec in db_postAC["P_wind"]}.values())
P_AC_hvdc_embedded = list({rec.keys[0]:rec.level for rec in db_postAC["P_hvdc_embedded"]}.values())
P_AC_hvdc_interconnection = list({rec.keys[0]:rec.level for rec in db_postAC["P_hvdc_interconnection"]}.values())
P_AC_hvdc_spit = list({rec.keys[0]:rec.level for rec in db_postAC["P_hvdc_spit"]}.values())
P_AC_dispatchable_load = list({rec.keys[0]:rec.level for rec in db_postAC["P_dispatchable_load"]}.values())

Q_AC_sync = list({rec.keys[0]:rec.level for rec in db_postAC["Q_sync"]}.values())
Q_AC_wind = list({rec.keys[0]:rec.level for rec in db_postAC["Q_wind"]}.values())
Q_AC_syncon = list({rec.keys[0]:rec.level for rec in db_postAC["Q_syncon"]}.values())
Q_AC_statcom = list({rec.keys[0]:rec.level for rec in db_postAC["Q_statcom"]}.values())
Q_AC_shunt = list({rec.keys[0]:rec.level for rec in db_postAC["Q_shunt"]}.values())
Q_AC_statvar = list({rec.keys[0]:rec.level for rec in db_postAC["Q_statvar"]}.values())
Q_AC_hvdc_embedded_1 = list({rec.keys[0]:rec.level for rec in db_postAC["Q_hvdc_embedded_1"]}.values())
Q_AC_hvdc_embedded_2 = list({rec.keys[0]:rec.level for rec in db_postAC["Q_hvdc_embedded_2"]}.values())
Q_AC_hvdc_interconnection = list({rec.keys[0]:rec.level for rec in db_postAC["Q_hvdc_interconnection"]}.values())
Q_AC_hvdc_spit = list({rec.keys[0]:rec.level for rec in db_postAC["Q_hvdc_spit"]}.values())

V_AC = list({rec.keys[0]:rec.level for rec in db_postAC["V"]}.values())
theta_AC = list({rec.keys[0]:rec.level for rec in db_postAC["theta"]}.values())

Q_penalty = db_postAC["Q_penalty"].first_record().level

print('Total cost:', round(cost, 2))
print('Sync gen:', sum(P_AC_sync) * baseMVA / 1000, ' / ', sum(sync_max) * baseMVA / 1000, '(commited) GW')
print('Wind gen:', (sum(P_AC_wind) + sum(P_AC_hvdc_spit)) * baseMVA / 1000, ' / ', (sum(wind_max) + spit_total_max) * baseMVA / 1000, 'GW')
print('Direct imports:', sum(list(P_AC_hvdc_interconnection)[:-1]) * baseMVA / 1000, ' / ', sum(hvdc_interconnection_max[:-1]) * baseMVA / 1000, 'GW (negative if exporting)')
print('Imports via England:', sum(list(P_AC_hvdc_interconnection)[-1:]) * baseMVA / 1000, ' / ', sum(hvdc_interconnection_max[-1:]) * baseMVA / 1000, 'GW (negative if exporting)')

total_generation = sum(P_AC_sync) + sum(P_AC_wind) + sum(P_AC_hvdc_spit)
total_demand = sum(demand_bus) - sum(P_AC_hvdc_interconnection)+ sum(P_AC_dispatchable_load)
print('Total generation:', total_generation * baseMVA / 1000, 'GW')
print('Generation/load imbalance (transmission losses in AC approx):', (total_generation - total_demand) * baseMVA / 1000, 'GW')
print('Price-responsive load:', sum(P_AC_dispatchable_load) * baseMVA / 1000, ' / ', sum(dispatchable_load_max) * baseMVA / 1000, 'GW')
print('Total embedded HVDC flows (North-South):', -sum(P_AC_hvdc_embedded) * baseMVA / 1000, ' / ', sum(hvdc_embedded_max) * baseMVA / 1000, 'GW')

def send_data_to_pf(sync_on, P_sync, P_wind, P_hvdc_embedded, P_hvdc_interconnection, P_hvdc_spit, P_dispatchable_load, V, Q_sync, Q_wind, Q_syncon, Q_statcom, Q_shunt, Q_statvar, Q_hvdc_embedded_1, Q_hvdc_embedded_2, Q_hvdc_interconnection, Q_hvdc_spit):
    for i, sync_gen in enumerate(sync_gens):
        sync_gen.pgini = P_sync[i] * baseMVA / sync_gen.ngnum
        sync_gen.qgini = Q_sync[i] * baseMVA / sync_gen.ngnum
        sync_gen.usetp = V[bus_names.index(sync_gen.bus1.GetParent().loc_name)]
        sync_gen.av_mode = 'constv'
        if sync_on[i] == 0:
            sync_gen.outServPzero = 1  # Do not use outserv as it is used to know which generators actually exist in the model
        else:
            sync_gen.outServPzero = 0

    for i, wind_gen in enumerate(wind_gens):
        wind_gen.pgini = P_wind[i] * baseMVA / wind_gen.ngnum
        wind_gen.qgini = Q_wind[i] * baseMVA / wind_gen.ngnum
        wind_gen.usetp = V[bus_names.index(wind_gen.bus1.GetParent().loc_name)]
        wind_gen.av_mode = 'constv'

    for i, syncon in enumerate(syncons):
        syncon.qgini = Q_syncon[i] * baseMVA / syncon.ngnum
        syncon.usetp = V[bus_names.index(syncon.bus1.GetParent().loc_name)]
        syncon.av_mode = 'constv'

    for i, statcom in enumerate(statcoms):
        statcom.qsetp = Q_statcom[i] * baseMVA
        statcom.iacdc = 6  # V control
        statcom.usetp = V[bus_names.index(statcom.busac.GetParent().loc_name)]

    for i, shunt in enumerate(shunts):
        shunt.ncapa = int(shunt.ncapx * (abs(Q_shunt[i]) / shunt_abs_max[i]))

    for i, statvar in enumerate(statvars):
        statvar.qsetp = Q_statvar[i] * baseMVA
        statvar.i_ctrl = 1  # V control
        statvar.usetp = V[bus_names.index(statvar.bus1.GetParent().loc_name)]

    for i, hvdc_embedded in enumerate(hvdc_embedded_1):
        hvdc_embedded.pgini = P_hvdc_embedded[i] * baseMVA / hvdc_embedded.ngnum
        hvdc_embedded.qgini = Q_hvdc_embedded_1[i] * baseMVA / hvdc_embedded.ngnum
        hvdc_embedded.usetp = V[bus_names.index(hvdc_embedded.bus1.GetParent().loc_name)]
        hvdc_embedded.av_mode = 'constv'
    find_by_loc_name(loads_incl_hvdc, 'HVDC WC load').plini = -P_hvdc_embedded[-1] * baseMVA  # Minus for generator to receptor convention
    find_by_loc_name(loads_incl_hvdc, 'HVDC NET EMBED').plini = P_hvdc_embedded[-1] * baseMVA * 0.98
    for i, hvdc_embedded in enumerate(hvdc_embedded_2):
        hvdc_embedded.pgini = P_hvdc_embedded[i] * baseMVA / hvdc_embedded.ngnum * (-0.98)  # Overwritten by Quasi-steady-state control anyway
        hvdc_embedded.qgini = Q_hvdc_embedded_2[i] * baseMVA / hvdc_embedded.ngnum
        hvdc_embedded.usetp = V[bus_names.index(hvdc_embedded.bus1.GetParent().loc_name)]
        hvdc_embedded.av_mode = 'constv'

    for i, hvdc_interconnection in enumerate(hvdc_interconnections):
        hvdc_interconnection.pgini = P_hvdc_interconnection[i] * baseMVA / hvdc_interconnection.ngnum
        hvdc_interconnection.qgini = Q_hvdc_interconnection[i] * baseMVA / hvdc_interconnection.ngnum
        hvdc_interconnection.usetp = V[bus_names.index(hvdc_interconnection.bus1.GetParent().loc_name)]
        hvdc_interconnection.av_mode = 'constv'
    find_by_loc_name(loads_incl_hvdc, 'HVDC NET IC').plini = -P_hvdc_interconnection[-1] * baseMVA  # Minus for generator to receptor convention

    for i, hvdc in enumerate(hvdc_spit):
        hvdc.pgini = P_hvdc_spit[i] * baseMVA / hvdc.ngnum
        hvdc.qgini = Q_hvdc_spit[i] * baseMVA / hvdc.ngnum
        hvdc.usetp = V[bus_names.index(hvdc.bus1.GetParent().loc_name)]
        hvdc.av_mode = 'constv'

    for i, load in enumerate(dispatchable_loads):
        load.plini = P_dispatchable_load[i] * baseMVA


def run_load_flow(allow_err=False):
    load_flow = app.GetFromStudyCase("ComLdf")
    load_flow.iopt_net = 0  # Balanced AC load flow
    load_flow.iPST_at = 0
    load_flow.iopt_plim = 0
    load_flow.iopt_at = 0
    load_flow.iopt_asht = 0
    load_flow.iopt_lim = 1  # Consider reactive power limits
    load_flow.iopt_pq = 0
    i_err = load_flow.Execute()  #ierr is the return value of the load flow calculation. 0 successful, 1 inner loop problem, 2 outer loop problem
    if i_err and not allow_err:
        raise RuntimeError("Load flow did not converge")
    return i_err

send_data_to_pf(on_DC, P_AC_sync, P_AC_wind, P_AC_hvdc_embedded, P_AC_hvdc_interconnection, P_AC_hvdc_spit, P_AC_dispatchable_load, V_AC, Q_AC_sync, Q_AC_wind, Q_AC_syncon,
                Q_AC_statcom, Q_AC_shunt, Q_AC_statvar, Q_AC_hvdc_embedded_1, Q_AC_hvdc_embedded_2, Q_AC_hvdc_interconnection, Q_AC_hvdc_spit)

#send_data_to_pf(on_DC, P_DC_sync, P_DC_wind, P_DC_hvdc_embedded, P_DC_hvdc_interconnection, P_DC_hvdc_spit, P_DC_dispatchable_load, V_AC, Q_AC_sync, Q_AC_wind, Q_AC_syncon,  # Check with DC values
#                Q_AC_statcom, Q_AC_shunt, Q_AC_statvar, Q_AC_hvdc_embedded_1, Q_AC_hvdc_embedded_2, Q_AC_hvdc_interconnection, Q_AC_hvdc_spit)

run_load_flow()

slack_imbalance_MW = find_by_loc_name(sync_gens, 'SG HC SLACK NGET4').GetAttribute('m:Psum:bus1')
if slack_imbalance_MW > 500:
    raise RuntimeError("Slack imbalance is too high, investigate", slack_imbalance_MW, 'MW')
print('Slack imbalance:', slack_imbalance_MW, 'MW')


#####
# PSCACOPF
#####
print('\nPSCACOPF')

# Initialise PSCACOPF based on results of ACOPF
P_PSCAC_sync = P_AC_sync.copy()
P_PSCAC_wind = P_AC_wind.copy()
P_PSCAC_hvdc_embedded = P_AC_hvdc_embedded.copy()
P_PSCAC_hvdc_interconnection = P_AC_hvdc_interconnection.copy()
P_PSCAC_hvdc_spit = P_AC_hvdc_spit.copy()
P_PSCAC_dispatchable_load = P_AC_dispatchable_load.copy()
V_PSCAC = V_AC.copy()
theta_PSCAC = theta_AC.copy()
Q_PSCAC_sync = Q_AC_sync.copy()
Q_PSCAC_wind = Q_AC_wind.copy()
Q_PSCAC_syncon = Q_AC_syncon.copy()
Q_PSCAC_statcom = Q_AC_statcom.copy()
Q_PSCAC_shunt = Q_AC_shunt.copy()
Q_PSCAC_statvar = Q_AC_statvar.copy()
Q_PSCAC_hvdc_embedded_1 = Q_AC_hvdc_embedded_1.copy()
Q_PSCAC_hvdc_embedded_2 = Q_AC_hvdc_embedded_2.copy()
Q_PSCAC_hvdc_interconnection = Q_AC_hvdc_interconnection.copy()
Q_PSCAC_hvdc_spit = Q_AC_hvdc_spit.copy()

boundary_B4_flow_max = 999
boundary_B6_flow_max = 999

critical_contingencies = []  # Contingencies that lead to issues and have to be included in the PSCACOPF (iteratively added to the problem)
while True:
    # Read values from load flow to initialise optimisation problem
    run_load_flow()

    P1 = np.zeros(N_branches)
    Q1 = np.zeros(N_branches)
    P2 = np.zeros(N_branches)
    Q2 = np.zeros(N_branches)
    boundary_B4_flow = 0
    boundary_B6_flow = 0
    index = 0
    for line in lines:
        P1[index] = line.GetAttribute('m:P:bus1') / baseMVA
        Q1[index] = line.GetAttribute('m:Q:bus1') / baseMVA
        P2[index] = line.GetAttribute('m:P:bus2') / baseMVA
        Q2[index] = line.GetAttribute('m:Q:bus2') / baseMVA
        if line in boundary_B4.branches:
            boundary_B4_flow += abs(line.GetAttribute('m:P:bus1') / baseMVA)
        elif line in boundary_B6.branches:
            boundary_B6_flow += abs(line.GetAttribute('m:P:bus1') / baseMVA)
        index += 1
    for tfo in tfos:
        P1[index] = tfo.GetAttribute('m:P:bushv') / baseMVA
        Q1[index] = tfo.GetAttribute('m:Q:bushv') / baseMVA
        P2[index] = tfo.GetAttribute('m:P:buslv') / baseMVA
        Q2[index] = tfo.GetAttribute('m:Q:buslv') / baseMVA
        index += 1
    for breaker in breakers:
        P1[index] = breaker.GetAttribute('m:P:bus1') / baseMVA
        Q1[index] = breaker.GetAttribute('m:Q:bus1') / baseMVA
        P2[index] = breaker.GetAttribute('m:P:bus2') / baseMVA
        Q2[index] = breaker.GetAttribute('m:Q:bus2') / baseMVA
        index += 1
    for series_capa in series_capas:
        P1[index] = series_capa.GetAttribute('m:P:bus1') / baseMVA
        Q1[index] = series_capa.GetAttribute('m:Q:bus1') / baseMVA
        P2[index] = series_capa.GetAttribute('m:P:bus2') / baseMVA
        Q2[index] = series_capa.GetAttribute('m:Q:bus2') / baseMVA
        index += 1

    P1_cont = np.zeros((N_branches, N_lines))
    Q1_cont = np.zeros((N_branches, N_lines))
    P2_cont = np.zeros((N_branches, N_lines))
    Q2_cont = np.zeros((N_branches, N_lines))
    Q_sync_cont = np.zeros((N_sync_gens, N_lines))
    Q_wind_cont = np.zeros((N_wind_gens, N_lines))
    Q_syncon_cont = np.zeros((N_syncons, N_lines))
    Q_statcom_cont = np.zeros((N_statcoms, N_lines))
    # Q_shunt_cont = np.zeros((N_shunts, N_lines))
    Q_statvar_cont = np.zeros((N_statvars, N_lines))
    Q_hvdc_embedded_1_cont = np.zeros((N_hvdc_embedded, N_lines))
    Q_hvdc_embedded_2_cont = np.zeros((N_hvdc_embedded, N_lines))
    Q_hvdc_interconnection_cont = np.zeros((N_hvdc_interconnections, N_lines))
    Q_hvdc_spit_cont = np.zeros((N_hvdc_spit, N_lines))
    V_cont = np.ones((N_buses, N_branches))
    V_dev_pos_cont = np.zeros((N_buses, N_branches))
    V_dev_neg_cont = np.zeros((N_buses, N_branches))
    theta_cont = np.zeros((N_buses, N_branches))

    # Compute power flows for each N-1 contingency
    current_critical_contingencies = []
    for j in range(N_lines):
        lines[j].outserv = 1

        err = run_load_flow(allow_err=True)
        if err:
            print('Load flow did not converge for contingency of line', lines[j].loc_name)
            current_critical_contingencies.append(j)
            continue

        critical = False
        index = 0
        for i, line in enumerate(lines):
            if i == j:
                P1_cont[index, j] = 0  # No flow in disconnected line
                Q1_cont[index, j] = 0
                P2_cont[index, j] = 0
                Q2_cont[index, j] = 0
            else:
                P1_cont[index, j] = line.GetAttribute('m:P:bus1') / baseMVA
                Q1_cont[index, j] = line.GetAttribute('m:Q:bus1') / baseMVA
                P2_cont[index, j] = line.GetAttribute('m:P:bus2') / baseMVA
                Q2_cont[index, j] = line.GetAttribute('m:Q:bus2') / baseMVA
                if line.GetAttribute('m:I1:bus1') > line.Inom_a:
                    critical = True
                    print('Overloaded line', line.loc_name, 'for contingency of line', lines[j].loc_name)
            index += 1
        for tfo in tfos:
            P1_cont[index, j] = tfo.GetAttribute('m:P:bushv') / baseMVA
            Q1_cont[index, j] = tfo.GetAttribute('m:Q:bushv') / baseMVA
            P2_cont[index, j] = tfo.GetAttribute('m:P:buslv') / baseMVA
            Q2_cont[index, j] = tfo.GetAttribute('m:Q:buslv') / baseMVA
            if tfo.GetAttribute('m:S:bushv') > tfo.typ_id.strn * tfo.ntnum:
                critical = True
                print('Overloaded tfo', tfo.loc_name, 'for contingency of line', lines[j].loc_name)
            index += 1
        for breaker in breakers:
            P1_cont[index, j] = breaker.GetAttribute('m:P:bus1') / baseMVA
            Q1_cont[index, j] = breaker.GetAttribute('m:Q:bus1') / baseMVA
            P2_cont[index, j] = breaker.GetAttribute('m:P:bus2') / baseMVA
            Q2_cont[index, j] = breaker.GetAttribute('m:Q:bus2') / baseMVA
            index += 1
        for series_capa in series_capas:
            P1_cont[index, j] = series_capa.GetAttribute('m:P:bus1') / baseMVA
            Q1_cont[index, j] = series_capa.GetAttribute('m:Q:bus1') / baseMVA
            P2_cont[index, j] = series_capa.GetAttribute('m:P:bus2') / baseMVA
            Q2_cont[index, j] = series_capa.GetAttribute('m:Q:bus2') / baseMVA
            index += 1

        for i, bus in enumerate(buses):
            V = bus.GetAttribute('m:u1')
            V_cont[i, j] = V
            theta_cont[i, j] = bus.GetAttribute('m:phiu') * pi/180 # theta_PSCAC[i]
            if V < V_PSCAC[i]:
                V_dev_neg_cont[i, j] = V_PSCAC[i] - V
            else:
                V_dev_pos_cont[i, j] = V - V_PSCAC[i]

            if V == 0:
                pass  # Isolated buses from the start or buses that cannot be N-1 secured
            elif V < 0.95:
                critical = True
                print('Undervoltage at bus', bus.loc_name, V, 'for contingency of line', lines[j].loc_name)
            elif V > 1.1:
                critical = True
                print('Overvoltage at bus', bus.loc_name, V, 'for contingency of line', lines[j].loc_name)

        if critical:
            current_critical_contingencies.append(j)

        for i in range(N_sync_gens):
            Q_sync_cont[i][j] = sync_gens[i].GetAttribute('m:Q:bus1') / baseMVA
        for i in range(N_wind_gens):
            Q_wind_cont[i][j] = wind_gens[i].GetAttribute('m:Q:bus1') / baseMVA
        for i in range(N_syncons):
            Q_syncon_cont[i][j] = syncons[i].GetAttribute('m:Q:bus1') / baseMVA
        for i in range(N_statcoms):
            Q_statcom_cont[i][j] = statcoms[i].GetAttribute('m:Q:busac') / baseMVA
        # for i in range(N_shunts):
        #     Q_shunt_cont[i][j] = shunts[i].GetAttribute('m:Q:bus1') / baseMVA
        for i in range(N_statvars):
            Q_statvar_cont[i][j] = statvars[i].GetAttribute('m:Q:bus1') / baseMVA
        for i in range(N_hvdc_embedded - 1):  # Assume 0 for Wester interconnector
            Q_hvdc_embedded_1_cont[i][j] = hvdc_embedded_1[i].GetAttribute('m:Q:bus1') / baseMVA
        for i in range(N_hvdc_embedded - 1):
            Q_hvdc_embedded_2_cont[i][j] = hvdc_embedded_2[i].GetAttribute('m:Q:bus1') / baseMVA
        for i in range(N_hvdc_interconnections - 1):  # Assume 0 for hvdc interconnectors through England
            Q_hvdc_interconnection_cont[i][j] = hvdc_interconnections[i].GetAttribute('m:Q:bus1') / baseMVA
        for i in range(N_hvdc_spit):
            Q_hvdc_spit_cont[i][j] = hvdc_spit[i].GetAttribute('m:Q:bus1') / baseMVA

        # Put back line in service
        lines[j].outserv = 0

    if set(current_critical_contingencies).issubset(critical_contingencies):  # No new critical contingencies compared to last iteration
        # Statically secured, so now check dynamic stability and limit flows through boundaries B4 and B6 if needed
        dynamic_secure = True

        B4_events = []
        if year == 2021:
            B4_events.append(['LOAN2_TUMM2_1', 'MELG4_DENN4_1'])  # Fault on parallel lines
            B4_events.append(['KINT2_LOAN2', 'LOAN2_TEAL_R2_2'])
            B4_events.append(['LOAN2_TEAL2_R1_1', 'LOAN2_TEAL2_R1_2'])
        elif year == 2030:
            B4_events.append(['KINA4_DENN4_1', 'MELG4_DENN4_1'])
            B4_events.append(['ALYT4_KINC4_1', 'ALYT4_KINC4_2'])
            B4_events.append(['LOAN2_TEAL2_R1_1', 'LOAN2_TEAL2_R1_2'])
        else:
            raise NotImplementedError('Year not considered')
        B6_events = []
        B6_events.append(['HARK4_ELVA4_1', 'HARK4_ELVA4_2'])
        B6_events.append(['STEW4_ECCL4_1', 'STEW4_ECCL4_2'])

        t_end = 5
        init = app.GetFromStudyCase("ComInc")
        init.iopt_reinc = 2  # Always reinitialise algebraic equations at interuption, seems to significantly improve numerical stability
        init.i_sedirect = 1  # DSL: direct application of events, significantly improve computation time
        simu = app.GetFromStudyCase("ComSim")
        simu.tstop = t_end
        faultType = 0  # 3-Phase short circuit
        faultTime = 0
        clearTime = 0.1

        B4_secure = True
        app.Show()
        for event in B4_events:
            print('Checking dynamic security for faults in B4 boundary', event)
            app.ResetCalculation()
            clearSimEvents()
            for line_name in event:
                line = find_by_loc_name(lines, line_name)
                add_short_circuit_event(line, faultTime, faultType, position=50)
                add_switch_event(line, clearTime, switch_action=0)  # switch_action: 0 = open, 1 = close
            simu.Execute()
            result_file = os.path.join(os.getcwd(), 'results.csv')
            export_results_to_csv(result_file)
            secure = check_stability(result_file, buses, sync_machines, t_end)
            if not secure:
                B4_secure = False
                dynamic_secure = False
                boundary_B4_flow_max = boundary_B4_flow - 100 / baseMVA
                if boundary_B4_flow_max < 0:
                    raise RuntimeError('Infeasible problem')
                break

        B6_secure = True
        for event in B6_events:
            print('Checking dynamic security for faults in B6 boundary', event)
            app.ResetCalculation()
            clearSimEvents()
            for line_name in event:
                line = find_by_loc_name(lines, line_name)
                add_short_circuit_event(line, faultTime, faultType, position=50)
                add_switch_event(line, clearTime, switch_action=0)  # switch_action: 0 = open, 1 = close
            simu.Execute()
            result_file = os.path.join(os.getcwd(), 'results.csv')
            export_results_to_csv(result_file)
            secure = check_stability(result_file, buses, sync_machines, t_end)
            if not secure:
                B6_secure = False
                dynamic_secure = False
                boundary_B6_flow_max = boundary_B6_flow - 100 / baseMVA
                if boundary_B6_flow_max < 0:
                    raise RuntimeError('Infeasible problem')
                break
        app.Hide()
        if dynamic_secure:   # System statically and dynamically secure, so stop
            break

    for j in current_critical_contingencies:
        if j not in critical_contingencies:
            critical_contingencies.append(j)

    print('Running PSCACOPF for contingencies of lines: ', [lines[i].loc_name for i in critical_contingencies])

    pscacopf_path = os.path.join('c-PSCACOPF') #, str(hour))
    Path(pscacopf_path).mkdir(parents=True, exist_ok=True)
    ws = gams.GamsWorkspace(working_directory=os.path.join(os.getcwd(), pscacopf_path), debug=gams.DebugLevel.Off)
    db_prePSCAC = ws.add_database()
    # shutil.copy(os.path.join('c-PSCACOPF', 'ipopt.opt'), pscacopf_path)

    i_sync = addGamsSet(db_prePSCAC, 'i_sync', 'sync generators', range(1, N_sync_gens + 1))
    i_wind = addGamsSet(db_prePSCAC, 'i_wind', 'wind generators', range(1, N_wind_gens + 1))
    i_syncon = addGamsSet(db_prePSCAC, 'i_syncon', 'syncon generators', range(1, N_syncons + 1))
    i_statcom = addGamsSet(db_prePSCAC, 'i_statcom', 'statcoms', range(1, N_statcoms + 1))
    i_shunt = addGamsSet(db_prePSCAC, 'i_shunt', 'shunts', range(1, N_shunts + 1))
    i_statvar = addGamsSet(db_prePSCAC, 'i_statvar', 'statvars', range(1, N_statvars + 1))
    i_bus = addGamsSet(db_prePSCAC, 'i_bus', 'buses', range(1, N_buses + 1))
    i_branch = addGamsSet(db_prePSCAC, 'i_branch', 'branches', range(1, N_branches + 1))
    i_hvdc_embedded = addGamsSet(db_prePSCAC, 'i_hvdc_embedded', 'embedded hvdc links', range(1, N_hvdc_embedded + 1))
    i_hvdc_embedded_Q = addGamsSet(db_prePSCAC, 'i_hvdc_embedded_Q', 'embedded hvdc links reactive control', range(1, N_hvdc_embedded_Q + 1))
    i_hvdc_interconnection = addGamsSet(db_prePSCAC, 'i_hvdc_interconnection', 'hvdc interconnections', range(1, N_hvdc_interconnections + 1))
    i_hvdc_spit = addGamsSet(db_prePSCAC, 'i_hvdc_spit', 'hvdc spits', range(1, N_hvdc_spit + 1))
    i_dispatchable_load = addGamsSet(db_prePSCAC, 'i_dispatchable_load', 'dispatchable loads', range(1, N_dispatchable_loads + 1))
    i_contingency = addGamsSet(db_prePSCAC, 'i_contingency', 'contingencies', range(1, 1 + len(critical_contingencies)))
    i_B4 = addGamsSet(db_prePSCAC, 'i_B4', 'B4 branches', range(1, 1 + len(boundary_B4.branches)))
    i_B6 = addGamsSet(db_prePSCAC, 'i_B6', 'B6 branches', range(1, 1 + len(boundary_B6.branches)))

    addGamsParams(db_prePSCAC, 'sync_map', 'sync generators map', [i_sync, i_bus], sync_gen_map)
    addGamsParams(db_prePSCAC, 'wind_map', 'wind generators map', [i_wind, i_bus], wind_gen_map)
    addGamsParams(db_prePSCAC, 'syncon_map', 'syncon generators map', [i_syncon, i_bus], syncon_map)
    addGamsParams(db_prePSCAC, 'statcom_map', 'statcom generators map', [i_statcom, i_bus], statcom_map)
    addGamsParams(db_prePSCAC, 'shunt_map', 'shunt generators map', [i_shunt, i_bus], shunt_map)
    addGamsParams(db_prePSCAC, 'statvar_map', 'statvar generators map', [i_statvar, i_bus], statvar_map)
    addGamsParams(db_prePSCAC, 'branch_map', 'branches map', [i_branch, i_bus], branch_map)
    addGamsParams(db_prePSCAC, 'B4_map', 'branches map', [i_branch, i_B4], B4_map)
    addGamsParams(db_prePSCAC, 'B6_map', 'branches map', [i_branch, i_B6], B6_map)
    addGamsParams(db_prePSCAC, 'hvdc_embedded_map', 'hvdc_embedded map', [i_hvdc_embedded, i_bus], hvdc_embedded_map)
    addGamsParams(db_prePSCAC, 'hvdc_embedded_map_Q1', 'hvdc_embedded_Q1 map', [i_hvdc_embedded_Q, i_bus], hvdc_embedded_map_Q1)
    addGamsParams(db_prePSCAC, 'hvdc_embedded_map_Q2', 'hvdc_embedded_Q2 map', [i_hvdc_embedded_Q, i_bus], hvdc_embedded_map_Q2)
    addGamsParams(db_prePSCAC, 'hvdc_interconnection_map', 'hvdc_interconnection map', [i_hvdc_interconnection, i_bus], hvdc_interconnection_map)
    addGamsParams(db_prePSCAC, 'hvdc_spit_map', 'hvdc_spit map', [i_hvdc_spit, i_bus], hvdc_spit_map)
    addGamsParams(db_prePSCAC, 'dispatchable_load_map', 'dispatchable_load map', [i_dispatchable_load, i_bus], dispatchable_load_map)

    addGamsParams(db_prePSCAC, 'droop', 'droop of synchronous generators', [i_sync], droop)
    addGamsParams(db_prePSCAC, 'sync_min', 'sync generator minimum generation', [i_sync], sync_min)
    addGamsParams(db_prePSCAC, 'sync_max', 'sync generator maximum generation', [i_sync], sync_max)
    addGamsParams(db_prePSCAC, 'sync_Qmin', 'sync generator minimum reactive generation', [i_sync], sync_Qmin)
    addGamsParams(db_prePSCAC, 'sync_Qmax', 'sync generator maximum reactive generation', [i_sync], sync_Qmax)
    addGamsParams(db_prePSCAC, 'wind_max', 'wind generator maximum generation', [i_wind], wind_max)
    addGamsParams(db_prePSCAC, 'wind_Qmin', 'wind generator minimum reactive power', [i_wind], wind_Qmin)
    addGamsParams(db_prePSCAC, 'wind_Qmax', 'wind generator maximum reactive power', [i_wind], wind_Qmax)
    addGamsParams(db_prePSCAC, 'syncon_Qmin', 'syncon minimum reactive generation', [i_syncon], syncon_Qmin)
    addGamsParams(db_prePSCAC, 'syncon_Qmax', 'syncon maximum reactive generation', [i_syncon], syncon_Qmax)
    addGamsParams(db_prePSCAC, 'statcom_Qmin', 'statcom minimum reactive generation', [i_statcom], statcom_Qmin)
    addGamsParams(db_prePSCAC, 'statcom_Qmax', 'statcom maximum reactive generation', [i_statcom], statcom_Qmax)
    addGamsParams(db_prePSCAC, 'shunt_Qmin', 'shunt minimum reactive generation', [i_shunt], shunt_Qmin)
    addGamsParams(db_prePSCAC, 'shunt_Qmax', 'shunt maximum reactive generation', [i_shunt], shunt_Qmax)
    addGamsParams(db_prePSCAC, 'statvar_Qmin', 'statvar minimum reactive generation', [i_statvar], statvar_Qmin)
    addGamsParams(db_prePSCAC, 'statvar_Qmax', 'statvar maximum reactive generation', [i_statvar], statvar_Qmax)
    addGamsParams(db_prePSCAC, 'hvdc_embedded_min', 'embedded hvdc minimum generation', [i_hvdc_embedded], hvdc_embedded_min)
    addGamsParams(db_prePSCAC, 'hvdc_embedded_max', 'embedded hvdc maximum generation', [i_hvdc_embedded], hvdc_embedded_max)
    addGamsParams(db_prePSCAC, 'hvdc_embedded_Qmin', 'embedded hvdc minimum reactive generation', [i_hvdc_embedded_Q], hvdc_embedded_Qmin)
    addGamsParams(db_prePSCAC, 'hvdc_embedded_Qmax', 'embedded hvdc maximum reactive generation', [i_hvdc_embedded_Q], hvdc_embedded_Qmax)
    addGamsParams(db_prePSCAC, 'hvdc_interconnection_min', 'interconnection hvdc minimum generation', [i_hvdc_interconnection], hvdc_interconnection_min)
    addGamsParams(db_prePSCAC, 'hvdc_interconnection_max', 'interconnection hvdc maximum generation', [i_hvdc_interconnection], hvdc_interconnection_max)
    addGamsParams(db_prePSCAC, 'hvdc_interconnection_Qmin', 'interconnection hvdc minimum reactive generation', [i_hvdc_interconnection], hvdc_interconnection_Qmin)
    addGamsParams(db_prePSCAC, 'hvdc_interconnection_Qmax', 'interconnection hvdc maximum reactive generation', [i_hvdc_interconnection], hvdc_interconnection_Qmax)
    addGamsParams(db_prePSCAC, 'hvdc_spit_min', 'spit hvdc minimum generation', [i_hvdc_spit], hvdc_spit_min)
    addGamsParams(db_prePSCAC, 'hvdc_spit_max', 'spit hvdc maximum generation', [i_hvdc_spit], hvdc_spit_max)
    db_prePSCAC.add_parameter("hvdc_spit_total_max", 0, 'available wind for spit hvdcs').add_record().value = spit_total_max
    addGamsParams(db_prePSCAC, 'hvdc_spit_Qmin', 'spit hvdc minimum reactive generation', [i_hvdc_spit], hvdc_spit_Qmin)
    addGamsParams(db_prePSCAC, 'hvdc_spit_Qmax', 'spit hvdc maximum reactive generation', [i_hvdc_spit], hvdc_spit_Qmax)
    addGamsParams(db_prePSCAC, 'dispatchable_load_min', 'dispatchable load minimum generation', [i_dispatchable_load], dispatchable_load_min)
    addGamsParams(db_prePSCAC, 'dispatchable_load_max', 'dispatchable load maximum generation', [i_dispatchable_load], dispatchable_load_max)

    addGamsParams(db_prePSCAC, 'Gff', 'line conductance (from-from)', [i_branch], G_branch_FromFrom)
    addGamsParams(db_prePSCAC, 'Bff', 'line susceptance (from-from)', [i_branch], B_branch_FromFrom)
    addGamsParams(db_prePSCAC, 'Gft', 'line conductance (from-to)', [i_branch], G_branch_FromTo)
    addGamsParams(db_prePSCAC, 'Bft', 'line susceptance (from-to)', [i_branch], B_branch_FromTo)
    addGamsParams(db_prePSCAC, 'Gtf', 'line conductance (to-from)', [i_branch], G_branch_ToFrom)
    addGamsParams(db_prePSCAC, 'Btf', 'line susceptance (to-from)', [i_branch], B_branch_ToFrom)
    addGamsParams(db_prePSCAC, 'Gtt', 'line conductance (to-to)', [i_branch], G_branch_ToTo)
    addGamsParams(db_prePSCAC, 'Btt', 'line susceptance (to-to)', [i_branch], B_branch_ToTo)
    addGamsParams(db_prePSCAC, 'branch_max_N', 'Normal branch max power', [i_branch], np.array(branch_p_max))
    addGamsParams(db_prePSCAC, 'branch_max_E', 'Emergency branch max power', [i_branch], np.array(branch_p_max))

    addGamsParams(db_prePSCAC, 'demand', 'demand at each bus', [i_bus], demand_bus)
    addGamsParams(db_prePSCAC, 'demandQ', 'reactive demand at each bus', [i_bus], demand_bus_Q)

    addGamsParams(db_prePSCAC, 'lincost', 'linear cost', [i_sync], sync_gen_costs)
    addGamsParams(db_prePSCAC, 'hvdc_interconnection_costs', 'import cost', [i_hvdc_interconnection], hvdc_interconnection_costs)
    addGamsParams(db_prePSCAC, 'dispatchable_load_costs', 'dispatchable load costs', [i_dispatchable_load], dispatchable_load_costs)

    addGamsParams(db_prePSCAC, 'P_sync_0', 'Initial sync outputs', [i_sync], list(P_PSCAC_sync))
    addGamsParams(db_prePSCAC, 'P_wind_0', 'Initial wind outputs', [i_wind], list(P_PSCAC_wind))
    addGamsParams(db_prePSCAC, 'P_hvdc_embedded_0', 'Initial embedded hvdc flows', [i_hvdc_embedded], list(P_PSCAC_hvdc_embedded))
    addGamsParams(db_prePSCAC, 'P_hvdc_interconnection_0', 'Initial hvdc interconection flows', [i_hvdc_interconnection], list(P_PSCAC_hvdc_interconnection))
    addGamsParams(db_prePSCAC, 'P_hvdc_spit_0', 'Initial hvdc spit flows', [i_hvdc_spit], list(P_PSCAC_hvdc_spit))
    addGamsParams(db_prePSCAC, 'P_dispatchable_load_0', 'Initial dispatchable load demand', [i_dispatchable_load], list(P_PSCAC_dispatchable_load))

    addGamsParams(db_prePSCAC, 'Q_sync_0', 'Initial sync reactive outputs', [i_sync], list(Q_PSCAC_sync))
    addGamsParams(db_prePSCAC, 'Q_wind_0', 'Initial wind reactive outputs', [i_wind], list(Q_PSCAC_wind))
    addGamsParams(db_prePSCAC, 'Q_syncon_0', 'Initial syncon reactive outputs', [i_syncon], list(Q_PSCAC_syncon))
    addGamsParams(db_prePSCAC, 'Q_statcom_0', 'Initial statcom reactive outputs', [i_statcom], list(Q_PSCAC_statcom))
    addGamsParams(db_prePSCAC, 'Q_shunt_0', 'Initial shunt reactive outputs', [i_shunt], list(Q_PSCAC_shunt))
    addGamsParams(db_prePSCAC, 'Q_statvar_0', 'Initial statvar reactive outputs', [i_statvar], list(Q_PSCAC_statvar))
    addGamsParams(db_prePSCAC, 'Q_hvdc_embedded_1_0', 'Initial hvdc_embedded_1 reactive outputs', [i_hvdc_embedded_Q], list(Q_PSCAC_hvdc_embedded_1))
    addGamsParams(db_prePSCAC, 'Q_hvdc_embedded_2_0', 'Initial hvdc_embedded_2 reactive outputs', [i_hvdc_embedded_Q], list(Q_PSCAC_hvdc_embedded_2))
    addGamsParams(db_prePSCAC, 'Q_hvdc_interconnection_0', 'Initial hvdc_interconnection reactive outputs', [i_hvdc_interconnection], list(Q_PSCAC_hvdc_interconnection))
    addGamsParams(db_prePSCAC, 'Q_hvdc_spit_0', 'Initial hvdc_spit reactive outputs', [i_hvdc_spit], list(Q_PSCAC_hvdc_spit))

    addGamsParams(db_prePSCAC, 'contingency_states', 'Line states in the considered contingencies', [i_branch, i_contingency], contingency_states[:, critical_contingencies])

    addGamsParams(db_prePSCAC, 'Q_sync_ck_0', 'Initial sync reactive outputs after contingency i', [i_sync, i_contingency], Q_sync_cont[:, critical_contingencies])
    addGamsParams(db_prePSCAC, 'Q_wind_ck_0', 'Initial wind reactive outputs after contingency i', [i_wind, i_contingency], Q_wind_cont[:, critical_contingencies])
    addGamsParams(db_prePSCAC, 'Q_syncon_ck_0', 'Initial syncon reactive outputs after contingency i', [i_syncon, i_contingency], Q_syncon_cont[:, critical_contingencies])
    addGamsParams(db_prePSCAC, 'Q_statcom_ck_0', 'Initial statcom reactive outputs after contingency i', [i_statcom, i_contingency], Q_statcom_cont[:, critical_contingencies])
    # addGamsParams(db_prePSCAC, 'Q_shunt_ck_0', 'Initial shunt reactive outputs after contingency i', [i_shunt, i_contingency], Q_shunt_cont[:, critical_contingencies])
    addGamsParams(db_prePSCAC, 'Q_statvar_ck_0', 'Initial statvar reactive outputs after contingency i', [i_statvar, i_contingency], Q_statvar_cont[:, critical_contingencies])
    addGamsParams(db_prePSCAC, 'Q_hvdc_embedded_1_ck_0', 'Initial hvdc_embedded_1 reactive outputs after contingency i', [i_hvdc_embedded_Q, i_contingency], Q_hvdc_embedded_1_cont[:, critical_contingencies])
    addGamsParams(db_prePSCAC, 'Q_hvdc_embedded_2_ck_0', 'Initial hvdc_embedded_2 reactive outputs after contingency i', [i_hvdc_embedded_Q, i_contingency], Q_hvdc_embedded_2_cont[:, critical_contingencies])
    addGamsParams(db_prePSCAC, 'Q_hvdc_interconnection_ck_0', 'Initial hvdc_interconnection reactive outputs after contingency i', [i_hvdc_interconnection, i_contingency], Q_hvdc_interconnection_cont[:, critical_contingencies])
    addGamsParams(db_prePSCAC, 'Q_hvdc_spit_ck_0', 'Initial hvdc_spit reactive outputs after contingency i', [i_hvdc_spit, i_contingency], Q_hvdc_spit_cont[:, critical_contingencies])

    addGamsParams(db_prePSCAC, 'V_0', 'Initial voltages', [i_bus], list(V_PSCAC))
    addGamsParams(db_prePSCAC, 'theta_0', 'Initial angles', [i_bus], list(theta_PSCAC))
    addGamsParams(db_prePSCAC, 'V_ck_0', 'Voltages after contingency i', [i_bus, i_contingency], V_cont[:, critical_contingencies])
    addGamsParams(db_prePSCAC, 'Vdev_pos_ck_0', 'Positive voltage deviation after contingency i', [i_bus, i_contingency], V_dev_pos_cont[:, critical_contingencies])
    addGamsParams(db_prePSCAC, 'Vdev_neg_ck_0', 'Negative voltage deviation after contingency i', [i_bus, i_contingency], V_dev_neg_cont[:, critical_contingencies])
    addGamsParams(db_prePSCAC, 'theta_ck_0', 'Angles after contingency i', [i_bus, i_contingency], theta_cont[:, critical_contingencies])

    addGamsParams(db_prePSCAC, 'P1_0', 'Initial active flows (from-to)', [i_branch], P1)
    addGamsParams(db_prePSCAC, 'Q1_0', 'Initial reactive flows (from-to)', [i_branch], Q1)
    addGamsParams(db_prePSCAC, 'P2_0', 'Initial active flows (to-from)', [i_branch], P2)
    addGamsParams(db_prePSCAC, 'Q2_0', 'Initial reactive flows (to-from)', [i_branch], Q2)

    db_prePSCAC.add_parameter("B4_flow_0", 0, 'Initial total flow through B4 boundary').add_record().value = boundary_B4_flow
    db_prePSCAC.add_parameter("B6_flow_0", 0, 'Initial total flow through B6 boundary').add_record().value = boundary_B6_flow
    db_prePSCAC.add_parameter("B4_flow_max", 0, 'Max total flow through B4 boundary').add_record().value = boundary_B4_flow_max
    db_prePSCAC.add_parameter("B6_flow_max", 0, 'Max total flow through B6 boundary').add_record().value = boundary_B6_flow_max

    addGamsParams(db_prePSCAC, 'P1_ck_0', 'Active flows (from-to) after contingency', [i_branch, i_contingency], P1_cont[:, critical_contingencies])
    addGamsParams(db_prePSCAC, 'Q1_ck_0', 'Reactive flows (from-to) after contingency', [i_branch, i_contingency], Q1_cont[:, critical_contingencies])
    addGamsParams(db_prePSCAC, 'P2_ck_0', 'Active flows (to-from) after contingency', [i_branch, i_contingency], P2_cont[:, critical_contingencies])
    addGamsParams(db_prePSCAC, 'Q2_ck_0', 'Reactive flows (to-from) after contingency', [i_branch, i_contingency], Q2_cont[:, critical_contingencies])

    db_prePSCAC.export('PrePSCACOPF.gdx')
    t = ws.add_job_from_file('PSCACOPF.gms')
    t.run()

    db_postPSCAC = ws.add_database_from_gdx("PostPSCACOPF.gdx")

    solve_status = int(db_postPSCAC["sol"].first_record().value)
    if solve_status != 1 and solve_status != 2 and solve_status != 7:
        raise RuntimeError('PSCACOPF: no solution found, error code:', solve_status)


    P_PSCAC_sync = list({rec.keys[0]:rec.level for rec in db_postPSCAC["P_sync"]}.values())
    P_PSCAC_wind = list({rec.keys[0]:rec.level for rec in db_postPSCAC["P_wind"]}.values())
    P_PSCAC_hvdc_embedded = list({rec.keys[0]:rec.level for rec in db_postPSCAC["P_hvdc_embedded"]}.values())
    P_PSCAC_hvdc_interconnection = list({rec.keys[0]:rec.level for rec in db_postPSCAC["P_hvdc_interconnection"]}.values())
    P_PSCAC_hvdc_spit = list({rec.keys[0]:rec.level for rec in db_postPSCAC["P_hvdc_spit"]}.values())
    P_PSCAC_dispatchable_load = list({rec.keys[0]:rec.level for rec in db_postPSCAC["P_dispatchable_load"]}.values())
    Q_PSCAC_sync = list({rec.keys[0]:rec.level for rec in db_postPSCAC["Q_sync"]}.values())
    Q_PSCAC_wind = list({rec.keys[0]:rec.level for rec in db_postPSCAC["Q_wind"]}.values())
    Q_PSCAC_syncon = list({rec.keys[0]:rec.level for rec in db_postPSCAC["Q_syncon"]}.values())
    Q_PSCAC_statcom = list({rec.keys[0]:rec.level for rec in db_postPSCAC["Q_statcom"]}.values())
    Q_PSCAC_shunt = list({rec.keys[0]:rec.level for rec in db_postPSCAC["Q_shunt"]}.values())
    Q_PSCAC_statvar = list({rec.keys[0]:rec.level for rec in db_postPSCAC["Q_statvar"]}.values())
    Q_PSCAC_hvdc_embedded_1 = list({rec.keys[0]:rec.level for rec in db_postPSCAC["Q_hvdc_embedded_1"]}.values())
    Q_PSCAC_hvdc_embedded_2 = list({rec.keys[0]:rec.level for rec in db_postPSCAC["Q_hvdc_embedded_2"]}.values())
    Q_PSCAC_hvdc_interconnection = list({rec.keys[0]:rec.level for rec in db_postPSCAC["Q_hvdc_interconnection"]}.values())
    Q_PSCAC_hvdc_spit = list({rec.keys[0]:rec.level for rec in db_postPSCAC["Q_hvdc_spit"]}.values())
    V_PSCAC = list({rec.keys[0]:rec.level for rec in db_postPSCAC["V"]}.values())
    theta_PSCAC = list({rec.keys[0]:rec.level for rec in db_postPSCAC["theta"]}.values())

    # cost = db_postPSCAC["cost"].first_record().level
    deviation = db_postPSCAC["deviation"].first_record().level

    send_data_to_pf(on_DC, P_PSCAC_sync, P_PSCAC_wind, P_PSCAC_hvdc_embedded, P_PSCAC_hvdc_interconnection, P_PSCAC_hvdc_spit, P_PSCAC_dispatchable_load, V_PSCAC, Q_PSCAC_sync, Q_PSCAC_wind, Q_PSCAC_syncon,
                Q_PSCAC_statcom, Q_PSCAC_shunt, Q_PSCAC_statvar, Q_PSCAC_hvdc_embedded_1, Q_PSCAC_hvdc_embedded_2, Q_PSCAC_hvdc_interconnection, Q_PSCAC_hvdc_spit)

    """
    # Disconnect PV generators at night
    for i in range(N_pv_gens):
        network.update_generators(id=pv_gens['GEN UID'][i], connected = pv_max[i] > 0)
    for i in range(N_rtpv_gens):
        network.update_generators(id=rtpv_gens['GEN UID'][i], connected = rtpv_max[i] > 0)

    for i in range(N_gens):
        bus_id = gens['Bus ID'][i]
        index = buses['Bus ID'].index(bus_id)
        V = list(V_PSCAC)[index] * buses['BaseKV'][index]
        network.update_generators(id=gens['GEN UID'][i], target_v=V)
    """

print('Total cost:', round(cost, 2))
print('Sync gen:', sum(P_PSCAC_sync) * baseMVA / 1000, ' / ', sum(sync_max) * baseMVA / 1000, '(commited) GW')
print('Wind gen:', (sum(P_PSCAC_wind) + sum(P_PSCAC_hvdc_spit)) * baseMVA / 1000, ' / ', (sum(wind_max) + spit_total_max) * baseMVA / 1000, 'GW')
print('Direct imports:', sum(list(P_PSCAC_hvdc_interconnection)[:-1]) * baseMVA / 1000, ' / ', sum(hvdc_interconnection_max[:-1]) * baseMVA / 1000, 'GW (negative if exporting)')
print('Imports via England:', sum(list(P_PSCAC_hvdc_interconnection)[-1:]) * baseMVA / 1000, ' / ', sum(hvdc_interconnection_max[-1:]) * baseMVA / 1000, 'GW (negative if exporting)')

total_generation = sum(P_PSCAC_sync) + sum(P_PSCAC_wind) + sum(P_PSCAC_hvdc_spit)
total_demand = sum(demand_bus) - sum(P_PSCAC_hvdc_interconnection)+ sum(P_PSCAC_dispatchable_load)
print('Total generation:', total_generation * baseMVA / 1000, 'GW')
print('Generation/load imbalance (transmission losses in AC approx):', (total_generation - total_demand) * baseMVA / 1000, 'GW')
print('Price-responsive load:', sum(P_PSCAC_dispatchable_load) * baseMVA / 1000, ' / ', sum(dispatchable_load_max) * baseMVA / 1000, 'GW')
print('Total embedded HVDC flows (North-South):', -sum(P_PSCAC_hvdc_embedded) * baseMVA / 1000, ' / ', sum(hvdc_embedded_max) * baseMVA / 1000, 'GW')

if len(current_critical_contingencies) > 0:
    print('Warning: remaining unsecured contingencies (diff between OPF and Powerfactory)', current_critical_contingencies)

run_load_flow()
app.Show()
slack_imbalance_MW = find_by_loc_name(sync_gens, 'SG HC SLACK NGET4').GetAttribute('m:Psum:bus1')
if slack_imbalance_MW > 500:
    raise RuntimeError("Slack imbalance is too high, investigate", slack_imbalance_MW, 'MW')
print('Slack imbalance:', slack_imbalance_MW, 'MW')
