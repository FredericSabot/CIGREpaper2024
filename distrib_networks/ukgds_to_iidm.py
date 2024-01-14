import pypowsybl as pp
import xlrd
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from lxml import etree
import subprocess
import merge_curves
from joblib import Parallel, delayed
import shutil
import random

NAMESPACE = 'http://www.rte-france.com/dynawo'
DYNAWO_PATH = '/home/fsabot/Desktop/dynawo_new/myEnvDynawo.sh'
DYNAWO_ALGO_PATH = '/home/fsabot/Desktop/dynawo-algorithms/myEnvDynawoAlgorithms.sh'

def get_row_values(sheet, row_idx):
    """
    Get row values from xlrd and cast them to the appropriate type
    """
    types = sheet.row_types(row_idx)
    values = sheet.row_values(row_idx)

    # Cast values according to their type
    return_row = []
    for i, value in enumerate(values):
        # Mapping of codes used by xlrd to types.
        type_codes = {
            0: "empty",
            1: "str",
            2: "float",
            3: "date",
            4: "bool",
            5: "error",
        }
        value_type = type_codes[types[i]]

        if value_type == "float":
            if value == int(value):
                value = int(value)
        # Format dates as a tuple.
        elif value_type == "date":
            value = xlrd.xldate_as_tuple(value, sheet.book.datemode)
        # Extract error text.
        elif value_type == "error":
            value = xlrd.error_text_from_code[value]
        return_row.append(value)
    return return_row


def add_substations_and_buses(network: pp.network.Network, book):
    # Read buses and transformers to identify substations
    bus_sheet = book.sheet_by_name('Buses')
    bus_ids = []
    for row_idx in range(29, bus_sheet.nrows):
        values = get_row_values(bus_sheet, row_idx)
        bus_ids.append(values[1])

    tfo_sheet = book.sheet_by_name('Transformers')
    tfo_from = []
    tfo_to = []
    for row_idx in range(29, tfo_sheet.nrows):
        values = get_row_values(tfo_sheet, row_idx)
        tfo_from.append(values[1])
        tfo_to.append(values[2])

    # Substations. Substations group busbars that are at the same location, 2 busbars connected by a transformer must be in the same substation in Powsybl
    bus_groups = []
    for i in range(len(tfo_from)):
        new_group = [tfo_from[i], tfo_to[i]]
        merge = False
        for bus_group in bus_groups:
            if new_group[0] in bus_group or new_group[1] in bus_group:
                # Merge new group to existing one with common bus
                merge = True
                for bus in new_group:
                    if bus not in bus_group:
                        bus_group.append(bus)
        if not merge:
            bus_groups.append(new_group)

    bus_to_substation_id = {}
    grouped_buses = []
    for bus_group in bus_groups:
        substation_id = min(bus_group)
        network.create_substations(id="S-{}".format(substation_id))
        for bus in bus_group:
            bus_to_substation_id[bus] = substation_id
            grouped_buses.append(bus)
    for bus_id in bus_ids:
        if bus_id not in grouped_buses:
            bus_to_substation_id[bus_id] = bus_id
            network.create_substations(id="S-{}".format(bus_id))

    # Write buses
    bus_sheet = book.sheet_by_name('Buses')
    for row_idx in range(29, bus_sheet.nrows):
        values = get_row_values(bus_sheet, row_idx)
        bus_id = values[1]
        bus_voltage = values[5]
        network.create_voltage_levels(id='VL-{}'.format(bus_id), substation_id='S-{}'.format(bus_to_substation_id[bus_id]), topology_kind='BUS_BREAKER',
                                    nominal_v=bus_voltage)
        network.create_buses(id='B-{}'.format(bus_id), voltage_level_id='VL-{}'.format(bus_id))


def add_lines(network: pp.network.Network, book, baseMVA):
    line_sheet = book.sheet_by_name('Branches')
    vl = network.get_voltage_levels()
    for row_idx in range(29, line_sheet.nrows):
        values = get_row_values(line_sheet, row_idx)
        bus_from = values[1]
        bus_to = values[2]
        line_id = values[3]
        line_r = values[4]
        line_x = values[5]
        line_b = values[6] / 2

        Ub = vl.at['VL-{}'.format(bus_from), 'nominal_v']
        Zb = Ub**2 / baseMVA

        network.create_lines(id='L-{}-{}_{}'.format(bus_from, bus_to, line_id), voltage_level1_id='VL-{}'.format(bus_from), bus1_id='B-{}'.format(bus_from),
                                    voltage_level2_id='VL-{}'.format(bus_to), bus2_id='B-{}'.format(bus_to),
                                    g1=0, g2=0, b1=line_b/Zb, b2=line_b/Zb,
                                    r=line_r * Zb, x=line_x * Zb)


def add_transformers(network: pp.network.Network, network_name, book, baseMVA):
    vl = network.get_voltage_levels()
    tfo_sheet = book.sheet_by_name('Transformers')
    for row_idx in range(29, tfo_sheet.nrows):
        values = get_row_values(tfo_sheet, row_idx)
        bus_from = values[1]
        bus_to = values[2]
        tfo_id = values[3]
        tfo_id = 'TFO-{}-{}_{}'.format(bus_from, bus_to, tfo_id)
        tfo_r = values[4]
        tfo_x = values[5]

        Ub = vl.at['VL-{}'.format(bus_to), 'nominal_v']  # Values given in secondary base (i.e. bus_to) for Powsybl
        Zb = Ub**2 / baseMVA
        r = tfo_r * Zb
        x = tfo_x * Zb

        # rating = values[12]
        # print(bus_to, tfo_x / rating * 100)
        # if tfo_x / rating > 0.1:  # TFOs should have an impedance of atmost 0.2pu (in transformer base) (actually, mostly 0.25 and a few 0.5 in SP distribution data)
            # tfo_x = 0.1 * rating
            # x = tfo_x * Zb

        tap_max = values[17]
        tap_min = values[18]
        tap_nb = values[19]

        network.create_2_windings_transformers(id=tfo_id,
                                               voltage_level1_id='VL-{}'.format(bus_from), bus1_id='B-{}'.format(bus_from),
                                               voltage_level2_id='VL-{}'.format(bus_to), bus2_id='B-{}'.format(bus_to),
                                               rated_u1=vl.at['VL-{}'.format(bus_from), 'nominal_v'], rated_u2=Ub,
                                               r=r, x=x, g=0, b=0)

        target_voltage = 1
        if Ub == 11:
            target_voltage = 11.2/11
        if network_name in ['ehv1', 'ehv2']:
            pass
        if network_name == 'ehv2':
            if Ub == 66:
                target_voltage = 1.02
            # target_voltage = 1.05
        elif network_name == 'ehv1':
            if Ub > 11:
                target_voltage = 1.03

        rtc_df = pd.DataFrame.from_records(
            index='id',
            columns=['id', 'target_deadband', 'target_v', 'on_load', 'tap', 'regulating', 'regulated_side'],
            data=[(tfo_id, Ub*0.02, Ub*target_voltage, True, tap_nb-1, True, 'TWO')])  # AVC equipment at GSPs is applied to each transformer such that the transf ormer secondary voltage is maintained within a pre-def ined dead band of +/-2% of the nominal secondary voltage and ensures that the tap changers on each transf ormer remain in step.
        tap_data = []
        for i in range(tap_nb):
            rho = tap_min + i/(tap_nb-1) * (tap_max - tap_min)
            tap_data.append((tfo_id, 0, 0, 0, 0, rho))

        steps_df = pd.DataFrame.from_records(
            index='id',
            columns=['id', 'b', 'g', 'r', 'x', 'rho'],
            data=tap_data)
        network.create_ratio_tap_changers(rtc_df, steps_df)


def add_shunts(network: pp.network.Network, book, baseMVA):
    vl = network.get_voltage_levels()
    shunt_sheet = book.sheet_by_name('Shunts')
    for row_idx in range(29, shunt_sheet.nrows):
        values = get_row_values(shunt_sheet, row_idx)
        shunt_bus = values[1]
        if shunt_bus == '':
            break
        shunt_id = values[2]
        shunt_b = 1/values[4]  # Positive values means capacitive in both ukgds and iidm

        Ub = vl.at['VL-{}'.format(shunt_bus), 'nominal_v']
        Yb = baseMVA / Ub**2

        shunt_df = pd.DataFrame.from_records(
            index='id',
            columns=['id', 'model_type', 'section_count', 'voltage_level_id', 'bus_id'],
            data=[('Shunt-{}_{}'.format(shunt_bus, shunt_id), 'LINEAR', 1, 'VL-{}'.format(shunt_bus), 'B-{}'.format(shunt_bus))])
        model_df = pd.DataFrame.from_records(
            index='id',
            columns=['id', 'g_per_section', 'b_per_section', 'max_section_count'],
            data=[('Shunt-{}_{}'.format(shunt_bus, shunt_id), 0, shunt_b * Yb, 1)])
        network.create_shunt_compensators(shunt_df, model_df)


def add_loads(network: pp.network.Network, network_name, book, base_factor, dyd_root, par_root, seed = None):
    # Static data
    load_sheet = book.sheet_by_name('Loads')
    vl = network.get_voltage_levels()
    for row_idx in range(29, load_sheet.nrows):
        values = get_row_values(load_sheet, row_idx)
        load_bus = values[1]
        load_id = values[2]
        load_p = values[3] * base_factor
        load_q = values[4] * base_factor
        if load_p == 0 and load_q == 0:
            continue

        # Loads are added behind an impedance, so add a new bus to attach it to
        id = 'Load-{}_{}'.format(load_bus, load_id)
        network.create_substations(id="S-{}".format(id))
        Ub = vl.at['VL-{}'.format(load_bus), 'nominal_v']
        network.create_voltage_levels(id='VL-{}'.format(id), substation_id='S-{}'.format(id), topology_kind='BUS_BREAKER', nominal_v=Ub)
        network.create_buses(id='B-{}'.format(id), voltage_level_id='VL-{}'.format(id))

        bus_from = load_bus
        bus_to = id
        Zb = Ub**2 / max(load_p, load_q)  # Add impedance in pu of the load
        if network_name in ['ehv1', 'ehv2']:
            X = 0.05 * Zb
        elif network_name in ['ehv3', 'ehv4']:
            X = 0.03 * Zb
        elif network_name in ['ehv5', 'ehv6']:
            X = 0.01 * Zb
        else:
            raise NotImplementedError(network_name, 'not considered')
        line_id = 'LineLoad-{}'.format(id)
        network.create_lines(id=line_id, voltage_level1_id='VL-{}'.format(bus_from), bus1_id='B-{}'.format(bus_from),
                                    voltage_level2_id='VL-{}'.format(bus_to), bus2_id='B-{}'.format(bus_to),
                                    g1=0, g2=0, b1=0, b2=0, r=0, x=X)

        network.create_loads(id='L-{}'.format(id), voltage_level_id='VL-{}'.format(id),
                            bus_id='B-{}'.format(id), p0=load_p, q0=load_q)

    # Dynamic data
    loads = network.get_loads()

    if seed == None:
        motor_share = 0.34
    else:
        motor_share = [0.3, 0.34, 0.4][seed]

    if motor_share != 0:
        lib = 'LoadAlphaBetaThreeMotorFifthOrder'
    else:
        lib = 'LoadAlphaBeta'

    for loadID in loads.index:
        load_attrib = {'id': loadID, 'lib': lib, 'parFile': network_name + '.par', 'parId': 'LoadAlphaBetaMotor', 'staticId': loadID}
        load = etree.SubElement(dyd_root, etree.QName(NAMESPACE, 'blackBoxModel'), load_attrib)

        etree.SubElement(load, etree.QName(NAMESPACE, 'macroStaticRef'), {'id': 'LOAD'})
        etree.SubElement(dyd_root, etree.QName(NAMESPACE, 'macroConnect'), {'id1': loadID, 'id2': 'NETWORK', 'connector': 'LOAD-CONNECTOR'})

        if 'Motor' in  load_attrib['lib']:
            etree.SubElement(dyd_root, etree.QName(NAMESPACE, 'connect'), {'id1': loadID, 'var1': 'load_omegaRefPu', 'id2': 'OMEGA_REF', 'var2': 'omegaRef_0'})

    motor_par_set = etree.SubElement(par_root, etree.QName(NAMESPACE, 'set'), {'id' : 'LoadAlphaBetaMotor'})
    # Motor A, B, C parameters from https://www.nerc.com/comm/PC/LoadModelingTaskForceDL/Dynamic%20Load%20Modeling%20Tech%20Ref%202016-11-14%20-%20FINAL.PDF
    par_attribs = [
        {'type': 'DOUBLE', 'name': 'load_alpha', 'value': '2'},
        {'type': 'DOUBLE', 'name': 'load_beta', 'value': '2'},
        {'type': 'DOUBLE', 'name': 'load_Alpha', 'value': '2'},
        {'type': 'DOUBLE', 'name': 'load_Beta', 'value': '2'},
        {'type': 'DOUBLE', 'name': 'load_ActiveMotorShare_0_', 'value': str(17.13/34.48 * motor_share)},
        {'type': 'DOUBLE', 'name': 'load_RsPu_0_', 'value': '0.04'},
        {'type': 'DOUBLE', 'name': 'load_LsPu_0_', 'value': '1.8'},
        {'type': 'DOUBLE', 'name': 'load_LPPu_0_', 'value': '0.12'},
        {'type': 'DOUBLE', 'name': 'load_LPPPu_0_', 'value': '0.104'},
        {'type': 'DOUBLE', 'name': 'load_tP0_0_', 'value': '0.095'},
        {'type': 'DOUBLE', 'name': 'load_tPP0_0_', 'value': '0.0021'},
        {'type': 'DOUBLE', 'name': 'load_H_0_', 'value': '0.1'},
        {'type': 'DOUBLE', 'name': 'load_torqueExponent_0_', 'value': '0'},
        {'type': 'DOUBLE', 'name': 'load_Utrip1Pu_0_', 'value': '0.65'},
        {'type': 'DOUBLE', 'name': 'load_tTrip1Pu_0_', 'value': '0.1'},
        {'type': 'DOUBLE', 'name': 'load_shareTrip1Pu_0_', 'value': '0.2'},
        {'type': 'DOUBLE', 'name': 'load_Ureconnect1Pu_0_', 'value': '0.1'},
        {'type': 'DOUBLE', 'name': 'load_tReconnect1Pu_0_', 'value': '9999'},
        {'type': 'DOUBLE', 'name': 'load_Utrip2Pu_0_', 'value': '0.5'},
        {'type': 'DOUBLE', 'name': 'load_tTrip2Pu_0_', 'value': '0.02'},
        {'type': 'DOUBLE', 'name': 'load_shareTrip2Pu_0_', 'value': '0.75'},
        {'type': 'DOUBLE', 'name': 'load_Ureconnect2Pu_0_', 'value': '0.65'},
        {'type': 'DOUBLE', 'name': 'load_tReconnect2Pu_0_', 'value': '0.1'},
        {'type': 'DOUBLE', 'name': 'load_ActiveMotorShare_1_', 'value': str(11.15/34.48 * motor_share)},
        {'type': 'DOUBLE', 'name': 'load_RsPu_1_', 'value': '0.03'},
        {'type': 'DOUBLE', 'name': 'load_LsPu_1_', 'value': '1.8'},
        {'type': 'DOUBLE', 'name': 'load_LPPu_1_', 'value': '0.19'},
        {'type': 'DOUBLE', 'name': 'load_LPPPu_1_', 'value': '0.14'},
        {'type': 'DOUBLE', 'name': 'load_tP0_1_', 'value': '0.2'},
        {'type': 'DOUBLE', 'name': 'load_tPP0_1_', 'value': '0.0026'},
        {'type': 'DOUBLE', 'name': 'load_H_1_', 'value': '0.5'},
        {'type': 'DOUBLE', 'name': 'load_torqueExponent_1_', 'value': '2'},
        {'type': 'DOUBLE', 'name': 'load_Utrip1Pu_1_', 'value': '0.55'},
        {'type': 'DOUBLE', 'name': 'load_tTrip1Pu_1_', 'value': '0.02'},
        {'type': 'DOUBLE', 'name': 'load_shareTrip1Pu_1_', 'value': '0.3'},
        {'type': 'DOUBLE', 'name': 'load_Ureconnect1Pu_1_', 'value': '0.65'},
        {'type': 'DOUBLE', 'name': 'load_tReconnect1Pu_1_', 'value': '0.05'},
        {'type': 'DOUBLE', 'name': 'load_Utrip2Pu_1_', 'value': '0.5'},
        {'type': 'DOUBLE', 'name': 'load_tTrip2Pu_1_', 'value': '0.025'},
        {'type': 'DOUBLE', 'name': 'load_shareTrip2Pu_1_', 'value': '0.3'},
        {'type': 'DOUBLE', 'name': 'load_Ureconnect2Pu_1_', 'value': '0.60'},
        {'type': 'DOUBLE', 'name': 'load_tReconnect2Pu_1_', 'value': '0.05'},
        {'type': 'DOUBLE', 'name': 'load_ActiveMotorShare_2_', 'value': str(6.2/34.48 * motor_share)},
        {'type': 'DOUBLE', 'name': 'load_RsPu_2_', 'value': '0.03'},
        {'type': 'DOUBLE', 'name': 'load_LsPu_2_', 'value': '1.8'},
        {'type': 'DOUBLE', 'name': 'load_LPPu_2_', 'value': '0.19'},
        {'type': 'DOUBLE', 'name': 'load_LPPPu_2_', 'value': '0.14'},
        {'type': 'DOUBLE', 'name': 'load_tP0_2_', 'value': '0.2'},
        {'type': 'DOUBLE', 'name': 'load_tPP0_2_', 'value': '0.0026'},
        {'type': 'DOUBLE', 'name': 'load_H_2_', 'value': '0.1'},
        {'type': 'DOUBLE', 'name': 'load_torqueExponent_2_', 'value': '2'},
        {'type': 'DOUBLE', 'name': 'load_Utrip1Pu_2_', 'value': '0.58'},
        {'type': 'DOUBLE', 'name': 'load_tTrip1Pu_2_', 'value': '0.03'},
        {'type': 'DOUBLE', 'name': 'load_shareTrip1Pu_2_', 'value': '0.2'},
        {'type': 'DOUBLE', 'name': 'load_Ureconnect1Pu_2_', 'value': '0.68'},
        {'type': 'DOUBLE', 'name': 'load_tReconnect1Pu_2_', 'value': '0.05'},
        {'type': 'DOUBLE', 'name': 'load_Utrip2Pu_2_', 'value': '0.53'},
        {'type': 'DOUBLE', 'name': 'load_tTrip2Pu_2_', 'value': '0.03'},
        {'type': 'DOUBLE', 'name': 'load_shareTrip2Pu_2_', 'value': '0.3'},
        {'type': 'DOUBLE', 'name': 'load_Ureconnect2Pu_2_', 'value': '0.62'},
        {'type': 'DOUBLE', 'name': 'load_tReconnect2Pu_2_', 'value': '0.05'},
    ]
    for par_attrib in par_attribs:
        etree.SubElement(motor_par_set, etree.QName(NAMESPACE, 'par'), par_attrib)

    references = [
        {'name': 'load_P0Pu', 'origData': 'IIDM', 'origName': 'p_pu', 'type': 'DOUBLE'},
        {'name': 'load_Q0Pu', 'origData': 'IIDM', 'origName': 'q_pu', 'type': 'DOUBLE'},
        {'name': 'load_U0Pu', 'origData': 'IIDM', 'origName': 'v_pu', 'type': 'DOUBLE'},
        {'name': 'load_UPhase0', 'origData': 'IIDM', 'origName': 'angle_pu', 'type': 'DOUBLE'},
    ]
    for ref in references:
        etree.SubElement(motor_par_set, etree.QName(NAMESPACE, 'reference'), ref)


def update_loads(network: pp.network.Network, load_ratio):
    """
    Multiply the current load by load_ratio
    """
    loads = network.get_loads()
    for load in loads.index:
        p = loads.at[load, 'p0'] * load_ratio
        q = loads.at[load, 'q0'] * load_ratio
        network.update_loads(pd.DataFrame({'p0': p, 'q0': q}, index=[load]))


def add_interconnections(network: pp.network.Network, network_name, book):
    if network_name != 'ehv4':
        return  # Only ehv4 is dependent on interconnections (this is shown by the similar max power capacities of the "generators" in the excel file)
    raise RuntimeError("It's difficult to simulate networks with more than one connection to the supergrid, and even more to make equivalents")
    vl = network.get_voltage_levels()
    gen_sheet = book.sheet_by_name('Generators')
    for row_idx in range(30, gen_sheet.nrows):  # Skip first (i.e. 29) as it is the slack bus
        values = get_row_values(gen_sheet, row_idx)
        gen_bus = values[1]
        gen_id = values[2]
        Ub = vl.at['VL-{}'.format(gen_bus), 'nominal_v']
        network.create_generators(id='GEN-{}_{}'.format(gen_bus, gen_id), voltage_level_id='VL-{}'.format(gen_bus), bus_id='B-{}'.format(gen_bus),
                                target_p=0, target_v=Ub, voltage_regulator_on=True, min_p=-999999, max_p=999999)


def add_grid_supply(network: pp.network.Network, network_name, book, dyd_root, par_root):
    # Static data
    vl = network.get_voltage_levels()
    bus_sheet = book.sheet_by_name('Buses')
    slack_bus = get_row_values(bus_sheet, 29)[1]  # Slack bus is assumed to be the first in the list
    if get_row_values(bus_sheet, 29)[6] != 'Slack':
        raise NotImplementedError('Slack bus assumed to be the first')
    Ub = vl.at['VL-{}'.format(slack_bus), 'nominal_v']

    total_demand = sum(network.get_loads().p0)
    network.create_generators(id='GEN-slack', voltage_level_id='VL-{}'.format(slack_bus), bus_id='B-{}'.format(slack_bus),
                              target_p=total_demand, target_v=Ub, voltage_regulator_on=True, min_p=-999999, max_p=999999)
    # Slack load
    # network.create_loads(id='LOAD-slack', voltage_level_id='VL-{}'.format(slack_bus), bus_id='B-{}'.format(slack_bus),
    #                      p0=0, q0=0)

    # Dynamic data
    """ gen_attrib = {'id': 'GEN-slack', 'lib': 'InfiniteBusWithVariations', 'parFile': network_name + '.par', 'parId': 'GEN-slack', 'staticId': 'GEN-slack'}
    etree.SubElement(dyd_root, etree.QName(NAMESPACE, 'blackBoxModel'), gen_attrib)
    etree.SubElement(dyd_root, etree.QName(NAMESPACE, 'connect'), {'id1': 'GEN-slack', 'var1': 'infiniteBus_terminal', 'id2': 'NETWORK', 'var2': '@STATIC_ID@@NODE@_ACPIN'})
    etree.SubElement(dyd_root, etree.QName(NAMESPACE, 'connect'), {'id1': 'GEN-slack', 'var1': 'infiniteBus_omegaPu', 'id2': 'OMEGA_REF', 'var2': 'omega_grp_0_value'})

    gen_par_set = etree.SubElement(par_root, etree.QName(NAMESPACE, 'set'), {'id' : 'GEN-slack'})
    par_attribs = [
        {'type': 'DOUBLE', 'name': 'infiniteBus_UEvtPu', 'value': '0.6'},
        {'type': 'DOUBLE', 'name': 'infiniteBus_tUEvtStart', 'value': '1'},
        {'type': 'DOUBLE', 'name': 'infiniteBus_tUEvtEnd', 'value': '1.2'},
        # Not used
        {'type': 'DOUBLE', 'name': 'infiniteBus_omega0Pu', 'value': '1'},
        {'type': 'DOUBLE', 'name': 'infiniteBus_omegaEvtPu', 'value': '1'},
        {'type': 'DOUBLE', 'name': 'infiniteBus_tOmegaEvtStart', 'value': '999'},
        {'type': 'DOUBLE', 'name': 'infiniteBus_tOmegaEvtEnd', 'value': '999'},
    ]
    for par_attrib in par_attribs:
        etree.SubElement(gen_par_set, etree.QName(NAMESPACE, 'par'), par_attrib)

    references = [
        {'name': 'infiniteBus_U0Pu', 'origData': 'IIDM', 'origName': 'v_pu', 'type': 'DOUBLE'},
        {'name': 'infiniteBus_UPhase', 'origData': 'IIDM', 'origName': 'angle_pu', 'type': 'DOUBLE'},
    ]
    for ref in references:
        etree.SubElement(gen_par_set, etree.QName(NAMESPACE, 'reference'), ref) """

    gen_attrib = {'id': 'GEN-slack', 'lib': 'GeneratorSynchronousFourWindingsProportionalRegulations', 'parFile': network_name + '.par', 'parId': 'GEN-slack', 'staticId': 'GEN-slack'}
    gen = etree.SubElement(dyd_root, etree.QName(NAMESPACE, 'blackBoxModel'), gen_attrib)
    etree.SubElement(dyd_root, etree.QName(NAMESPACE, 'macroConnect'), {'id1': 'GEN-slack', 'id2': 'OMEGA_REF', 'connector': 'MS_OMEGAREF_CONNECTOR', 'index2': '0'})
    etree.SubElement(gen, etree.QName(NAMESPACE, 'macroStaticRef'), {'id': 'GEN'})
    etree.SubElement(dyd_root, etree.QName(NAMESPACE, 'macroConnect'), {'id1': 'GEN-slack', 'id2': 'NETWORK', 'connector': 'GEN-CONNECTOR'})

    gen_par_set = etree.SubElement(par_root, etree.QName(NAMESPACE, 'set'), {'id' : 'GEN-slack'})
    par_attribs = [
        {'type': 'INT', 'name': 'generator_ExcitationPu', 'value': '1'},
        {'type': 'BOOL', 'name': 'generator_UseApproximation', 'value': 'true'},
        {'type': 'DOUBLE', 'name': 'generator_md', 'value': '0.215'},
        {'type': 'DOUBLE', 'name': 'generator_mq', 'value': '0.215'},
        {'type': 'DOUBLE', 'name': 'generator_nd', 'value': '6.995'},
        {'type': 'DOUBLE', 'name': 'generator_nq', 'value': '6.995'},
        {'type': 'DOUBLE', 'name': 'generator_MdPuEfd', 'value': '0'},
        {'type': 'DOUBLE', 'name': 'generator_DPu', 'value': '0'},
        {'type': 'DOUBLE', 'name': 'generator_H', 'value': '50'},
        {'type': 'DOUBLE', 'name': 'generator_RaPu', 'value': '0.0028'},
        {'type': 'DOUBLE', 'name': 'generator_XlPu', 'value': '0.202'},
        {'type': 'DOUBLE', 'name': 'generator_XdPu', 'value': '2.22'},
        {'type': 'DOUBLE', 'name': 'generator_XpdPu', 'value': '0.384'},
        {'type': 'DOUBLE', 'name': 'generator_XppdPu', 'value': '0.264'},
        {'type': 'DOUBLE', 'name': 'generator_Tpd0', 'value': '8.094'},
        {'type': 'DOUBLE', 'name': 'generator_Tppd0', 'value': '0.08'},
        {'type': 'DOUBLE', 'name': 'generator_XqPu', 'value': '2.22'},
        {'type': 'DOUBLE', 'name': 'generator_XpqPu', 'value': '0.393'},
        {'type': 'DOUBLE', 'name': 'generator_XppqPu', 'value': '0.262'},
        {'type': 'DOUBLE', 'name': 'generator_Tpq0', 'value': '1.572'},
        {'type': 'DOUBLE', 'name': 'generator_Tppq0', 'value': '0.084'},
        {'type': 'DOUBLE', 'name': 'generator_UNom', 'value': '1'},
        {'type': 'DOUBLE', 'name': 'generator_SNom', 'value': str(total_demand * 3.3)},
        {'type': 'DOUBLE', 'name': 'generator_PNomTurb', 'value': str(total_demand * 3)},
        {'type': 'DOUBLE', 'name': 'generator_PNomAlt', 'value': str(total_demand * 3)},
        {'type': 'DOUBLE', 'name': 'generator_SnTfo', 'value': str(total_demand * 3.3)},
        {'type': 'DOUBLE', 'name': 'generator_UNomHV', 'value': '1'},
        {'type': 'DOUBLE', 'name': 'generator_UNomLV', 'value': '1'},
        {'type': 'DOUBLE', 'name': 'generator_UBaseHV', 'value': '1'},
        {'type': 'DOUBLE', 'name': 'generator_UBaseLV', 'value': '1'},
        {'type': 'DOUBLE', 'name': 'generator_RTfPu', 'value': '0.0'},
        {'type': 'DOUBLE', 'name': 'generator_XTfPu', 'value': str(0.2 * 3)},  # Represents 0.1 for the TFO and 0.1 for a line between the TFO and slack bus
        {'type': 'DOUBLE', 'name': 'voltageRegulator_LagEfdMax', 'value': '0'},
        {'type': 'DOUBLE', 'name': 'voltageRegulator_LagEfdMin', 'value': '0'},
        {'type': 'DOUBLE', 'name': 'voltageRegulator_EfdMinPu', 'value': '-5'},
        {'type': 'DOUBLE', 'name': 'voltageRegulator_EfdMaxPu', 'value': '5'},
        {'type': 'DOUBLE', 'name': 'voltageRegulator_UsRefMinPu', 'value': '0.8'},
        {'type': 'DOUBLE', 'name': 'voltageRegulator_UsRefMaxPu', 'value': '1.2'},
        {'type': 'DOUBLE', 'name': 'voltageRegulator_Gain', 'value': '20'},
        {'type': 'DOUBLE', 'name': 'governor_KGover', 'value': '5'},
        {'type': 'DOUBLE', 'name': 'governor_PMin', 'value': '0'},
        {'type': 'DOUBLE', 'name': 'governor_PMax', 'value': str(total_demand * 3)},
        {'type': 'DOUBLE', 'name': 'governor_PNom', 'value': str(total_demand * 3)},
    ]
    for par_attrib in par_attribs:
        etree.SubElement(gen_par_set, etree.QName(NAMESPACE, 'par'), par_attrib)

    references = [
        {'name': 'generator_P0Pu', 'origData': 'IIDM', 'origName': 'p_pu', 'type': 'DOUBLE'},
        {'name': 'generator_Q0Pu', 'origData': 'IIDM', 'origName': 'q_pu', 'type': 'DOUBLE'},
        {'name': 'generator_U0Pu', 'origData': 'IIDM', 'origName': 'v_pu', 'type': 'DOUBLE'},
        {'name': 'generator_UPhase0', 'origData': 'IIDM', 'origName': 'angle_pu', 'type': 'DOUBLE'},
    ]
    for ref in references:
        etree.SubElement(gen_par_set, etree.QName(NAMESPACE, 'reference'), ref)

    # Omegaref
    omega_attrib = {'id': 'OMEGA_REF', 'lib': 'DYNModelOmegaRef', 'parFile': network_name + '.par', 'parId': 'OmegaRef'}
    etree.SubElement(dyd_root, etree.QName(NAMESPACE, 'blackBoxModel'), omega_attrib)
    omega_par_set = etree.SubElement(par_root, etree.QName(NAMESPACE, 'set'), {'id' : 'OmegaRef'})
    etree.SubElement(omega_par_set, etree.QName(NAMESPACE, 'par'), {'type': 'DOUBLE', 'name': 'weight_gen_' + str(0), 'value': str(1)})
    etree.SubElement(omega_par_set, etree.QName(NAMESPACE, 'par'), {'type': 'INT', 'name': 'nbGen', 'value': str(1)})  # str(omega_index)})


def write_disturbance_files(network: pp.network.Network, network_name, book, par_root, output_path):
    """
    Write the fic_MULTIPLE.xml and the associated .dyd file for each considered disturbance
    """
    bus_sheet = book.sheet_by_name('Buses')
    slack_bus = get_row_values(bus_sheet, 29)[1]  # Slack bus is assumed to be the first in the list
    total_demand = sum(network.get_loads().p0)

    disturbances = [
        (0.2, 0.1),  # Voltage dip (pu), duration (s)
        (0.2, 0.2),
        (0.3, 0.1),
        (0.3, 0.2),
        (0.4, 0.1),
        (0.4, 0.2),
        (0.5, 0.1),
        (0.5, 0.2),
        (0.7, 0.1),
        (0.7, 0.2),
        (0.8, 0.2),
        (0.8, 0.5),
        (0.9, 0.5),
        (0.9, 1.0),
    ]

    for i, disturbance in enumerate(disturbances):
        i += 1  # 1-indexing
        root = etree.Element(etree.QName(NAMESPACE, 'dynamicModelsArchitecture'), nsmap={'dyn': NAMESPACE})
        fault_attrib = {'id': 'Fault', 'lib': 'NodeFault', 'parFile': network_name + '.par', 'parId': 'Fault_{}'.format(i)}
        etree.SubElement(root, etree.QName(NAMESPACE, 'blackBoxModel'), fault_attrib)
        connect_attrib = {'id1': 'Fault', 'var1': 'fault_terminal', 'id2': 'NETWORK', 'var2': 'B-{}_ACPIN'.format(slack_bus)}
        etree.SubElement(root, etree.QName(NAMESPACE, 'connect'), connect_attrib)
        with open(output_path + 'Fault_{}.dyd'.format(i), 'wb') as doc:
            doc.write(etree.tostring(root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))

        fault_par_set = etree.SubElement(par_root, etree.QName(NAMESPACE, 'set'), {'id' : 'Fault_{}'.format(i)})
        V_fault = disturbance[0]
        par_attribs = [
            {'type': 'DOUBLE', 'name': 'fault_RPu', 'value': '0'},
            # Voltage divider: V_fault = X_fault / (0.2 + X_fault) (neglects load impedance) -> X_fault = 0.2 * V_fault / (1 + V_fault)
            {'type': 'DOUBLE', 'name': 'fault_XPu', 'value': str(0.2/(total_demand/100) * (V_fault/(1-V_fault)))},
            {'type': 'DOUBLE', 'name': 'fault_tBegin', 'value': '1'},
            {'type': 'DOUBLE', 'name': 'fault_tEnd', 'value': str(1 + disturbance[1])},
        ]
        for par_attrib in par_attribs:
            etree.SubElement(fault_par_set, etree.QName(NAMESPACE, 'par'), par_attrib)

        with open(output_path + '_Fault_{}.dyd'.format(i), 'wb') as doc:
            doc.write(etree.tostring(root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))

    # fic_MULTIPLE.xml
    fic_root = etree.Element(etree.QName(NAMESPACE, 'multipleJobs'), nsmap={'fic': NAMESPACE})
    scenarios = etree.SubElement(fic_root, etree.QName(NAMESPACE, 'scenarios'), {'jobsFile': network_name + '.jobs'})
    for i in range(1, len(disturbances) + 1):
        etree.SubElement(scenarios, etree.QName(NAMESPACE, 'scenario'), {'id': 'Fault_{}'.format(i), 'dydFile': network_name + 'Fault_{}.dyd'.format(i)})

    output_dir = os.path.dirname(output_path)
    with open(os.path.join(output_dir, network_name + '_fic.xml') , 'wb') as doc:
        doc.write(etree.tostring(fic_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))


def add_reactive_compensation(network: pp.network.Network, book, reactive_compensation):
    bus_sheet = book.sheet_by_name('Buses')
    slack_bus = get_row_values(bus_sheet, 29)[1]  # Slack bus is assumed to be the first in the list

    network.create_loads(id='L_Compensation', voltage_level_id='VL-{}'.format(slack_bus),
                        bus_id='B-{}'.format(slack_bus), p0=0, q0=reactive_compensation)


def add_distributed_energy_resources(network: pp.network.Network, network_name, book, base_factor, dyd_root, par_root, der_installed_share, der_capacity_factor, der_legacy_share, random_generator = None, der_pf=1):
    # Put distributed generation in parallel with all loads
    if der_installed_share == 0:
        return
    load_sheet = book.sheet_by_name('Loads')
    for row_idx in range(29, load_sheet.nrows):
        values = get_row_values(load_sheet, row_idx)
        der_bus = values[1]

        for typ in ['legacy', 'G99']:
            der_p_max = values[3] * base_factor * der_installed_share
            if typ == 'legacy':
                der_p_max *= der_legacy_share
            elif typ == 'G99':
                der_p_max *= (1-der_legacy_share)
            der_p = der_p_max * der_capacity_factor
            der_q = der_p * (1-der_pf**2)**0.5
            if der_p == 0 and der_q == 0:
                continue

            der_id = values[2]
            load_id = 'Load-{}_{}'.format(der_bus, der_id)  # ID of the load to which the DER is added in parallel to
            der_id = 'DER-{}_{}_{}'.format(der_bus, der_id, typ)
            network.create_generators(id=der_id, voltage_level_id='VL-{}'.format(load_id), bus_id='B-{}'.format(load_id),
                                target_p=der_p, target_q=der_q, voltage_regulator_on=False, min_p=0, max_p=der_p_max)

            # Dynamic data
            gen_attrib = {'id': der_id, 'lib': 'der_a_GenericLVRT', 'parFile': network_name + '.par', 'parId': der_id, 'staticId': der_id}
            gen = etree.SubElement(dyd_root, etree.QName(NAMESPACE, 'blackBoxModel'), gen_attrib)
            etree.SubElement(gen, etree.QName(NAMESPACE, 'macroStaticRef'), {'id': 'PV'})
            etree.SubElement(dyd_root, etree.QName(NAMESPACE, 'macroConnect'), {'id1': der_id, 'id2': 'NETWORK', 'connector': 'PV-CONNECTOR'})
            etree.SubElement(dyd_root, etree.QName(NAMESPACE, 'connect'), {'id1': der_id, 'var1': 'ibg_omegaRefPu', 'id2': 'OMEGA_REF', 'var2': 'omegaRef_0_value'})

            gen_par_set = etree.SubElement(par_root, etree.QName(NAMESPACE, 'set'), {'id' : der_id})

            if typ == 'legacy':
                par_attribs = [
                    {'type': 'BOOL', 'name': 'ibg_PPriority', 'value': 'true'},
                    {'type': 'DOUBLE', 'name': 'ibg_KQsupportPu', 'value': '0'},
                    {'type': 'DOUBLE', 'name': 'ibg_ULVRTMinPu', 'value': '0.87'},
                    {'type': 'DOUBLE', 'name': 'ibg_ULVRTIntPu', 'value': '0.8'},
                    {'type': 'DOUBLE', 'name': 'ibg_ULVRTArmingPu', 'value': '0.8'},
                    {'type': 'DOUBLE', 'name': 'ibg_tLVRTMin', 'value': '1'},
                    {'type': 'DOUBLE', 'name': 'ibg_tLVRTInt', 'value': '1'},
                    {'type': 'DOUBLE', 'name': 'ibg_tLVRTMax', 'value': '2.5'},
                ]
            elif typ == 'G99':
                par_attribs = [
                    {'type': 'BOOL', 'name': 'ibg_PPriority', 'value': 'false'},
                    {'type': 'DOUBLE', 'name': 'ibg_KQsupportPu', 'value': '2.5'},
                    {'type': 'DOUBLE', 'name': 'ibg_ULVRTMinPu', 'value': '0.1'},
                    {'type': 'DOUBLE', 'name': 'ibg_ULVRTIntPu', 'value': '0.1'},
                    {'type': 'DOUBLE', 'name': 'ibg_ULVRTArmingPu', 'value': '0.85'},
                    {'type': 'DOUBLE', 'name': 'ibg_tLVRTMin', 'value': '0.14'},
                    {'type': 'DOUBLE', 'name': 'ibg_tLVRTInt', 'value': '0.14'},
                    {'type': 'DOUBLE', 'name': 'ibg_tLVRTMax', 'value': '2.2'},
                ]
            else:
                raise NotImplementedError()
            par_attribs += [
                {'type': 'DOUBLE', 'name': 'ibg_IMaxPu', 'value': '1.1'},
                {'type': 'DOUBLE', 'name': 'ibg_tFilterU', 'value': '0.02'},
                {'type': 'DOUBLE', 'name': 'ibg_RecoveringShare', 'value': '0'},
                {'type': 'DOUBLE', 'name': 'ibg_tFilterOmega', 'value': '0.02'},
                {'type': 'DOUBLE', 'name': 'ibg_tP', 'value': '0.02'},
                {'type': 'DOUBLE', 'name': 'ibg_tG', 'value': '0.02'},
                {'type': 'DOUBLE', 'name': 'ibg_tPord', 'value': '0.02'},
                {'type': 'DOUBLE', 'name': 'ibg_tIq', 'value': '0.02'},
                {'type': 'DOUBLE', 'name': 'ibg_fDeadZoneMaxPu', 'value': '0.4'},
                {'type': 'DOUBLE', 'name': 'ibg_fDeadZoneMinPu', 'value': '-1'},
                {'type': 'DOUBLE', 'name': 'ibg_DdnPu', 'value': '10'},
                {'type': 'DOUBLE', 'name': 'ibg_DupPu', 'value': '0'},
                {'type': 'DOUBLE', 'name': 'ibg_feMaxPu', 'value': '99'},
                {'type': 'DOUBLE', 'name': 'ibg_feMinPu', 'value': '-99'},
                {'type': 'DOUBLE', 'name': 'ibg_Kpg', 'value': '0.1'},
                {'type': 'DOUBLE', 'name': 'ibg_Kig', 'value': '10'},
                {'type': 'DOUBLE', 'name': 'ibg_PMaxPu', 'value': '1'},
                {'type': 'DOUBLE', 'name': 'ibg_PMinPu', 'value': '0'},
                {'type': 'DOUBLE', 'name': 'ibg_DPMaxPu', 'value': '0.5'},
                {'type': 'DOUBLE', 'name': 'ibg_DPMinPu', 'value': '-0.5'},
                {'type': 'BOOL', 'name': 'ibg_Freq_flag', 'value': 'true'},
                {'type': 'DOUBLE', 'name': 'ibg_OmegaMaxPu', 'value': '1.05'},
                {'type': 'DOUBLE', 'name': 'ibg_OmegaMinPu', 'value': '0.95'},
                {'type': 'DOUBLE', 'name': 'ibg_tOmegaMaxPu', 'value': '3'},
                {'type': 'DOUBLE', 'name': 'ibg_tOmegaMinPu', 'value': '3'},
                {'type': 'DOUBLE', 'name': 'ibg_IpRateLimMax', 'value': '2.5'},
                {'type': 'DOUBLE', 'name': 'ibg_IpRateLimMin', 'value': '-999'},
                {'type': 'DOUBLE', 'name': 'ibg_VRefPu', 'value': '1'},
                {'type': 'DOUBLE', 'name': 'ibg_VDeadzoneMaxPu', 'value': '0.1'},
                {'type': 'DOUBLE', 'name': 'ibg_VDeadzoneMinPu', 'value': '-0.1'},
                {'type': 'DOUBLE', 'name': 'ibg_iQSupportMaxPu', 'value': '999'},
                {'type': 'DOUBLE', 'name': 'ibg_iQSupportMinPu', 'value': '-999'},
            ]
            """
            {'type': 'DOUBLE', 'name': 'ibg_IMaxPu', 'value': '1.1'},
            {'type': 'DOUBLE', 'name': 'ibg_UPLLFreezePu', 'value': '0.1'},
            {'type': 'DOUBLE', 'name': 'ibg_UQPrioPu', 'value': '0.01'},
            {'type': 'DOUBLE', 'name': 'ibg_US1', 'value': '0.9'},
            {'type': 'DOUBLE', 'name': 'ibg_US2', 'value': '1.1'},
            {'type': 'DOUBLE', 'name': 'ibg_kRCI', 'value': '0'},  # 2.5
            {'type': 'DOUBLE', 'name': 'ibg_kRCA', 'value': '0'},
            {'type': 'DOUBLE', 'name': 'ibg_m', 'value': '0.1'},
            {'type': 'DOUBLE', 'name': 'ibg_n', 'value': '0.1'},
            {'type': 'DOUBLE', 'name': 'ibg_tG', 'value': '0.1'},
            {'type': 'DOUBLE', 'name': 'ibg_Tm', 'value': '0.1'},
            {'type': 'DOUBLE', 'name': 'ibg_IpSlewMaxPu', 'value': '1'},
            {'type': 'DOUBLE', 'name': 'ibg_IqSlewMaxPu', 'value': '5'},
            {'type': 'DOUBLE', 'name': 'ibg_tLVRTMin', 'value': '0.14'},
            {'type': 'DOUBLE', 'name': 'ibg_tLVRTInt', 'value': '0.14'},
            {'type': 'DOUBLE', 'name': 'ibg_tLVRTMax', 'value': '2.2'},
            {'type': 'DOUBLE', 'name': 'ibg_ULVRTArmingPu', 'value': '0.9'},
            {'type': 'DOUBLE', 'name': 'ibg_OmegaMaxPu', 'value': '1.05'},
            {'type': 'DOUBLE', 'name': 'ibg_OmegaDeadBandPu', 'value': '1.01'},
            {'type': 'DOUBLE', 'name': 'ibg_OmegaMinPu', 'value': '0.95'},
            {'type': 'DOUBLE', 'name': 'ibg_tFilterOmega', 'value': '0.1'},
            {'type': 'DOUBLE', 'name': 'ibg_tFilterU', 'value': '0.01'},
            {'type': 'DOUBLE', 'name': 'ibg_UMaxPu', 'value': '1.2'},
            {'type': 'DOUBLE', 'name': 'ibg_Kf', 'value': '0'},
            {'type': 'DOUBLE', 'name': 'ibg_tf', 'value': '0.1'},
            {'type': 'DOUBLE', 'name': 'ibg_PLLFreeze_Ki', 'value': '20'},
            {'type': 'DOUBLE', 'name': 'ibg_PLLFreeze_Kp', 'value': '3'},
            """
            """
            if random_generator:
                u_min = random_generator.uniform(0.05, 0.8)
                par_attribs += [
                    {'type': 'DOUBLE', 'name': 'ibg_ULVRTMinPu', 'value': str(u_min)},
                    {'type': 'DOUBLE', 'name': 'ibg_ULVRTIntPu', 'value': str(u_min)},
                ]
            else:
                par_attribs += [
                    {'type': 'DOUBLE', 'name': 'ibg_ULVRTMinPu', 'value': '0.8'},
                    {'type': 'DOUBLE', 'name': 'ibg_ULVRTIntPu', 'value': '0.8'},
                ]
            """
            for par_attrib in par_attribs:
                etree.SubElement(gen_par_set, etree.QName(NAMESPACE, 'par'), par_attrib)

            references = [
                {'name': 'ibg_SNom', 'origData': 'IIDM', 'origName': 'pMax', 'type': 'DOUBLE'},
                {'name': 'ibg_P0Pu', 'origData': 'IIDM', 'origName': 'p_pu', 'type': 'DOUBLE'},
                {'name': 'ibg_Q0Pu', 'origData': 'IIDM', 'origName': 'q_pu', 'type': 'DOUBLE'},
                {'name': 'ibg_U0Pu', 'origData': 'IIDM', 'origName': 'v_pu', 'type': 'DOUBLE'},
                {'name': 'ibg_UPhase0', 'origData': 'IIDM', 'origName': 'angle_pu', 'type': 'DOUBLE'},
            ]
            for ref in references:
                etree.SubElement(gen_par_set, etree.QName(NAMESPACE, 'reference'), ref)


def run_loadflow(network: pp.network.Network, lowest_regulating_transformer):
    """
    Run a load flow where only tap changers with a target voltage (in kV)
    higher than "lowest_regulating_transformer" are active. If "lowest_regulating_transformer" = 0, all tap changers are active.
    """
    parameters = pp.loadflow.Parameters(distributed_slack=False)
    def load_generation_balance(network):
        n_iter = 0
        slack_p = 0
        while n_iter < 10:
            lf_results = pp.loadflow.run_ac(network, parameters)
            if int(lf_results[0].status) != 0:
                raise Exception('Load flow did not converge', n_iter)

            delta_P = lf_results[0].slack_bus_active_power_mismatch
            total_demand = sum(network.get_loads().p0)  # Used to make results independent of base
            if (abs(delta_P) / total_demand < 1e-6):
                break
            slack_p += delta_P
            balance = {'target_p' : slack_p}
            network.update_generators(pd.DataFrame(balance, index=['GEN-slack']))
            n_iter += 1

    # Tap changers
    n_iter = 0
    current_lowest_regulating_transformer = 400
    while True:
        load_generation_balance(network)

        if current_lowest_regulating_transformer < lowest_regulating_transformer:
            break

        buses = network.get_buses()
        tap_changers = network.get_ratio_tap_changers()
        tap_change = False
        for tap_changer in tap_changers.index:
            if tap_changers.at[tap_changer, 'target_v'] < current_lowest_regulating_transformer:
                continue

            v = buses.at[tap_changers.at[tap_changer, 'regulating_bus_id'], 'v_mag']
            if v < tap_changers.at[tap_changer, 'target_v'] - tap_changers.at[tap_changer, 'target_deadband']:
                if tap_changers.at[tap_changer, 'tap'] < tap_changers.at[tap_changer, 'high_tap']:
                    tap_change = True
                    tap = tap_changers.at[tap_changer, 'tap'] + 1
                    network.update_ratio_tap_changers(pd.DataFrame({'tap' : tap}, index=[tap_changer]))
            elif v > tap_changers.at[tap_changer, 'target_v'] + tap_changers.at[tap_changer, 'target_deadband']:
                if tap_changers.at[tap_changer, 'tap'] > tap_changers.at[tap_changer, 'low_tap']:
                    tap_change = True
                    tap = tap_changers.at[tap_changer, 'tap'] - 1
                    network.update_ratio_tap_changers(pd.DataFrame({'tap' : tap}, index=[tap_changer]))
            else:
                pass

        if not tap_change:
            if current_lowest_regulating_transformer < min(tap_changers.target_v):
                break
            current_lowest_regulating_transformer /= 2  # First set the tap of the HV transformers, and progressively allow lower voltage transformer to be regulating too

        n_iter += 1
        if n_iter > 100:
            raise RuntimeError('Erratic behaviour of tap changers')



def show_voltage_profile(network: pp.network.Network, output_name, book):
    vl = network.get_voltage_levels()
    buses = network.get_buses()

    # Read bus_ids from excel instead of Pypowsybl because it renames buses for some reason
    bus_sheet = book.sheet_by_name('Buses')
    bus_ids = []
    for row_idx in range(29, bus_sheet.nrows):
        values = get_row_values(bus_sheet, row_idx)
        bus_ids.append(values[1])

    v_profile = []
    for bus_id in bus_ids:
        Ub = vl.at['VL-{}'.format(bus_id), 'nominal_v']
        U = buses.at['VL-{}_0'.format(bus_id), 'v_mag']  # Pypowsybl renames buses for some reason
        v_profile.append(U/Ub)

    ax = pd.Series(v_profile).plot(figsize=(16,9), kind='bar')  # plt.bar(x=range(len(bus_ids)), height=v_profile)
    ax.set_ylim(bottom=0.85, top=1.15)
    plt.axhline(1.05, color='red')
    plt.axhline(0.95, color='red')
    plt.axhline(1.02, color='red', alpha=0.5)
    plt.axhline(0.98, color='red', alpha=0.5)
    ax.set_xticklabels(bus_ids)
    plt.savefig(output_name + '.pdf')
    plt.close()


def write_crv_file(output_path, book, detailed=False):
    crv_root = etree.Element(etree.QName(NAMESPACE, 'curvesInput'))

    etree.SubElement(crv_root, etree.QName(NAMESPACE, 'curve'), {'model': 'GEN-slack', 'variable': 'infiniteBus_PPu'})
    etree.SubElement(crv_root, etree.QName(NAMESPACE, 'curve'), {'model': 'GEN-slack', 'variable': 'infiniteBus_QPu'})
    etree.SubElement(crv_root, etree.QName(NAMESPACE, 'curve'), {'model': 'GEN-slack', 'variable': 'generator_PGen'})
    etree.SubElement(crv_root, etree.QName(NAMESPACE, 'curve'), {'model': 'GEN-slack', 'variable': 'generator_QGen'})

    """ # Power in the main transformer
    tfo_sheet = book.sheet_by_name('Transformers')
    for row_idx in range(29, tfo_sheet.nrows):
        values = get_row_values(tfo_sheet, row_idx)
        bus_from = values[1]
        bus_to = values[2]
        tfo_id = values[3]
        tfo_id = 'TFO-{}-{}_{}'.format(bus_from, bus_to, tfo_id)
        etree.SubElement(crv_root, etree.QName(NAMESPACE, 'curve'), {'model': 'NETWORK', 'variable': tfo_id + '_P1_value'})
        etree.SubElement(crv_root, etree.QName(NAMESPACE, 'curve'), {'model': 'NETWORK', 'variable': tfo_id + '_Q1_value'})
        if not detailed:
            break  # Only shows curves for the first transformer (supposed to be connected to the GSP) """

    # Bus voltages
    if detailed:
        bus_sheet = book.sheet_by_name('Buses')
        for row_idx in range(29, bus_sheet.nrows):
            values = get_row_values(bus_sheet, row_idx)
            bus_id = 'B-{}'.format(values[1])
            etree.SubElement(crv_root, etree.QName(NAMESPACE, 'curve'), {'model': 'NETWORK', 'variable': bus_id + '_Upu_value'})

    with open(output_path + '.crv', 'wb') as doc:
        doc.write(etree.tostring(crv_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))


def dump_network_file(network: pp.network.Network, output_path):
    network_string = network.dump_to_string('XIIDM', {'iidm.export.xml.version' : '1.4'})
    network_string = network_string.encode()
    XMLparser = etree.XMLParser(remove_blank_text=True)
    root = etree.fromstring(network_string, XMLparser)
    root.set('sourceFormat', 'UKGDS')
    with open(output_path + '.iidm', 'wb') as doc:
        doc.write(etree.tostring(root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))
    # network.write_network_area_diagram_svg(file_name + '.svg')


def ukgds_to_dynawo(input_file, load_ratio, der_installed_share, der_capacity_factor, der_legacy_share, output_path=None, seed=None):
    random_generator = None
    if seed:
        random_generator = random.Random(seed)
    XMLparser = etree.XMLParser(remove_blank_text=True)  # Necessary for pretty_print to work
    dyd_root = etree.parse('base.dyd', XMLparser).getroot()
    par_root = etree.parse('base.par', XMLparser).getroot()

    network = pp.network.create_empty()
    book = xlrd.open_workbook(input_file)

    sys_sheet = book.sheet_by_name('System')
    baseMVA = sys_sheet.cell(29, 2).value
    intro_sheet = book.sheet_by_name('Introduction')
    max_gross_load = intro_sheet.cell(39, 2).value
    common_base = 100  # Note: results can be slightly impacted by the change of base due to numerical instabilities in the load flow (with tap changers)
    base_factor = common_base / max_gross_load  # Factor used such that all networks have the same max_gross_load

    network_name, ext = os.path.basename(input_file).rsplit('.', 1)

    add_substations_and_buses(network, book)
    add_lines(network, book, baseMVA=(baseMVA*base_factor))
    add_transformers(network, network_name, book, baseMVA=(baseMVA*base_factor))
    add_shunts(network, book, baseMVA=(baseMVA*base_factor))
    add_loads(network, network_name, book, base_factor, dyd_root, par_root, seed)
    add_interconnections(network, network_name, book)
    add_grid_supply(network, network_name, book, dyd_root, par_root)

    # Run a first load flow at peak load (and no DER) and with all tap changers active (used to define the tap of manual tap changers)
    run_loadflow(network, lowest_regulating_transformer=0)
    Path('figs').mkdir(parents=True, exist_ok=True)
    show_voltage_profile(network, 'figs/' + network_name + '_peak', book)

    update_loads(network, load_ratio)
    add_distributed_energy_resources(network, network_name, book, base_factor, dyd_root, par_root, der_installed_share, der_capacity_factor, der_legacy_share, random_generator)

    # Rerun loadflow with current load and DERs but constant tap settings for the low voltage transformers (manual tap changers)
    run_loadflow(network, lowest_regulating_transformer=25)  # 132/33kV and 33/11kV tap changers are all assumed to be automatic
    show_voltage_profile(network, 'figs/' + network_name + '_actual', book)

    # Add a reactive load at the GSP to have a global power factor of 0, used to ease the comparison between the different ehv networks
    gens = network.get_generators()
    reactive_compensation = gens.at['GEN-slack', 'q']
    add_reactive_compensation(network, book, reactive_compensation)
    run_loadflow(network, lowest_regulating_transformer=25)

    write_disturbance_files(network, network_name, book, par_root, output_path)

    if output_path:
        dump_network_file(network, output_path)
        with open(output_path + '.dyd', 'wb') as doc:
            doc.write(etree.tostring(dyd_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))
        with open(output_path + '.par', 'wb') as doc:
            doc.write(etree.tostring(par_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))
        write_crv_file(output_path, book, detailed=True)
        shutil.copy(os.path.join('dynawo_files', 'jobs_files', network_name + '.jobs'), output_path + '.jobs')

    return network


def build_network_and_simulate(network_name, load_ratio, der_installed_share, der_capacity_factor, der_legacy_share, seed=None, rerun_simulations=True):
    input_file = os.path.join('ukgds', network_name + '.xls')
    if seed is not None:
        output_dir = os.path.join('dynawo_files', 'Random', str(seed))
    else:
        output_dir = os.path.join('dynawo_files', 'Deterministic')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    network_name, ext = os.path.basename(input_file).rsplit('.', 1)
    output_path = os.path.join(output_dir, network_name)

    faults = ['Fault_{}'.format(i) for i in range(1, 14 + 1)]
    output_curve_path = [os.path.join(output_dir, fault, network_name, 'curves', 'curves.csv') for fault in faults]
    fic_path = os.path.join(output_dir, network_name + '_fic.xml')
    if any(not Path(path).exists() for path in output_curve_path) or rerun_simulations:
        ukgds_to_dynawo(input_file, load_ratio, der_installed_share, der_capacity_factor, der_legacy_share, output_path, seed)

        # cmd = [DYNAWO_PATH, 'jobs', os.path.join(output_dir, network_name + '.jobs')]
        cmd = [DYNAWO_ALGO_PATH, 'SA', '--directory', output_dir, '--input', network_name + '_fic.xml', '--output', network_name + '_aggregatedResults.xml', '--nbThreads', '5']
        subprocess.run(cmd)

    return output_curve_path, network_name, fic_path


if __name__ == '__main__':
    RERUN_SIMULATIONS = True
    RANDOMISE = True
    load_ratio = 1
    der_installed_share = 0.8
    der_legacy_share = 0.5
    der_capacity_factor = 1

    if RANDOMISE:
        seeds = range(3)
    else:
        seeds = [None]
    network_names = ['ehv{}'.format(i) for i in range(1, 7) if i != 4]

    PARALLEL = False
    if PARALLEL:
        results = Parallel(n_jobs=5)(delayed(build_network_and_simulate)(network_name, load_ratio, der_installed_share, der_capacity_factor, der_legacy_share, seed, RERUN_SIMULATIONS) for network_name in network_names for seed in seeds)
    else:
        results = []
        for network_name in network_names:
            for seed in seeds:
                results.append(build_network_and_simulate(network_name, load_ratio, der_installed_share, der_capacity_factor, der_legacy_share, seed, RERUN_SIMULATIONS))

    output_curve_paths = []
    network_names = []
    for result in results:
        output_curve_paths.append(result[0])
        network_names.append(result[1])

    fig_name = os.path.join('figs', 'power' + '_load_' + str(load_ratio) + '_der_' + str(der_installed_share) + '_' + str(der_capacity_factor) + '_legacy_' + str(der_legacy_share) + '.pdf')
    merge_curves.plot_power(output_curve_paths, network_names, fig_name)
    fig_name = os.path.join('figs', 'voltage' + '_load_' + str(load_ratio) + '_der_' + str(der_installed_share) + '_' + str(der_capacity_factor) + '_legacy_' + str(der_legacy_share) + '.pdf')
    merge_curves.plot_voltages(output_curve_paths, network_names, fig_name)
