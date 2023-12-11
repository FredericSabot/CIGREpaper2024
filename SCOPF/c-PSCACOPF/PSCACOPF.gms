***************************************************************
*** SETS
***************************************************************

set i_sync sync generators;
set i_wind wind generators;
set i_syncon synchronous condensers;
set i_statcom statcoms;
set i_shunt shunt capacitors and reactors;
set i_statvar SVCs;
set i_bus buses;
set i_branch branches;
set i_hvdc_embedded embedded hvdc links;
set i_hvdc_embedded_Q embedded hvdc links reactive control;
set i_hvdc_interconnection hvdc interconnections;
set i_hvdc_spit hvdc spits;
set i_dispatchable_load dispatchable loads;
set i_contingency contingencies;
set i_B4 boundary B4 branches;
set i_B6 boundary B6 branches;

*needed for running twice through the same set in a single equation
alias(i_bus,j_bus);

***************************************************************
*** PARAMETERS
***************************************************************

parameter Epsilon;
Epsilon = 1e-4;

parameter Kg;
Kg = 0.1;


*GENERATOR DATA

parameter sync_map(i_sync, i_bus) sync generator map;
parameter wind_map(i_wind, i_bus) wind generator map;
parameter syncon_map(i_syncon, i_bus) SC map;
parameter statcom_map(i_statcom, i_bus) SC map;
parameter shunt_map(i_shunt, i_bus) SC map;
parameter statvar_map(i_statvar, i_bus) SC map;
parameter hvdc_embedded_map(i_hvdc_embedded, i_bus) hvdc_embedded map of links;
parameter hvdc_embedded_map_Q1(i_hvdc_embedded_Q, i_bus) hvdc_embedded map of converters side 1;
parameter hvdc_embedded_map_Q2(i_hvdc_embedded_Q, i_bus) hvdc_embedded map of converters side 2;
parameter hvdc_interconnection_map(i_hvdc_interconnection, i_bus) hvdc_interconnection map;
parameter hvdc_spit_map(i_hvdc_spit, i_bus) hvdc_spit map;
parameter dispatchable_load_map(i_dispatchable_load, i_bus) dispatchable_load map;

parameter lincost(i_sync) slope of each generator cost curve block;
parameter hvdc_interconnection_costs(i_hvdc_interconnection) slope of import costs;
parameter dispatchable_load_costs(i_dispatchable_load) gain of dispatchable loads;

parameter sync_min(i_sync) sync generator minimum generation;
parameter sync_max(i_sync) sync generator maximum generation;
parameter sync_Qmin(i_sync) sync generator minimum reactive generation;
parameter sync_Qmax(i_sync) sync generator maximum reactive generation;

parameter wind_max(i_wind) wind generator maximum generation;
parameter wind_Qmin(i_wind) wind generator minimum reactive power;
parameter wind_Qmax(i_wind) wind generator maximum reactive power;

parameter syncon_Qmin(i_syncon) syncon minimum reactive generation;
parameter syncon_Qmax(i_syncon) syncon maximum reactive generation;
parameter statcom_Qmin(i_statcom) statcom minimum reactive generation;
parameter statcom_Qmax(i_statcom) statcom maximum reactive generation;
parameter shunt_Qmin(i_shunt) shunt minimum reactive generation;
parameter shunt_Qmax(i_shunt) shunt maximum reactive generation;
parameter statvar_Qmin(i_statvar) statvar minimum reactive generation;
parameter statvar_Qmax(i_statvar) statvar maximum reactive generation;

parameter hvdc_embedded_min(i_hvdc_embedded) embedded hvdc minimum generation;
parameter hvdc_embedded_max(i_hvdc_embedded) embedded hvdc maximum generation;
parameter hvdc_embedded_Qmin(i_hvdc_embedded_Q) embedded hvdc minimum reactive generation;
parameter hvdc_embedded_Qmax(i_hvdc_embedded_Q) embedded hvdc maximum reactive generation;
parameter hvdc_interconnection_min(i_hvdc_interconnection) interconnection hvdc minimum generation;
parameter hvdc_interconnection_max(i_hvdc_interconnection) interconnection hvdc maximum generation;
parameter hvdc_interconnection_Qmin(i_hvdc_interconnection) interconnection hvdc minimum reactive generation;
parameter hvdc_interconnection_Qmax(i_hvdc_interconnection) interconnection hvdc maximum reactive generation;
parameter hvdc_spit_min(i_hvdc_spit) spit hvdc minimum generation;
parameter hvdc_spit_max(i_hvdc_spit) spit hvdc maximum generation;
parameter hvdc_spit_total_max available wind for spit hvdcs;
parameter dispatchable_load_min(i_dispatchable_load) interconnection hvdc minimum generation;
parameter dispatchable_load_max(i_dispatchable_load) interconnection hvdc maximum generation;
parameter hvdc_spit_Qmin(i_hvdc_spit) spit hvdc minimum reactive generation;
parameter hvdc_spit_Qmax(i_hvdc_spit) spit hvdc maximum reactive generation;

parameter P_sync_0(i_sync) initial sync outputs;
parameter P_wind_0(i_wind) initial wind outputs;
parameter P_hvdc_embedded_0(i_hvdc_embedded) initial embedded hvdc flows;
parameter P_hvdc_interconnection_0(i_hvdc_interconnection) initial hvdc interconnection flows;
parameter P_hvdc_spit_0(i_hvdc_spit) initial hvdc spit flows;
parameter P_dispatchable_load_0(i_dispatchable_load) initial hvdc interconnection flows;

parameter Q_sync_0(i_sync) Initial sync reactive outputs;
parameter Q_wind_0(i_wind) Initial wind reactive outputs;
parameter Q_syncon_0(i_syncon) Initial syncon reactive outputs;
parameter Q_statcom_0(i_statcom) Initial statcom reactive outputs;
parameter Q_shunt_0(i_shunt) Initial shunt reactive outputs;
parameter Q_statvar_0(i_statvar) Initial statvar reactive outputs;
parameter Q_hvdc_embedded_1_0(i_hvdc_embedded_Q) Initial hvdc_embedded_1 reactive outputs;
parameter Q_hvdc_embedded_2_0(i_hvdc_embedded_Q) Initial hvdc_embedded_2 reactive outputs;
parameter Q_hvdc_interconnection_0(i_hvdc_interconnection) Initial hvdc_interconnection reactive outputs;
parameter Q_hvdc_spit_0(i_hvdc_spit) Initial hvdc_spit reactive outputs;

parameter Q_sync_ck_0(i_sync, i_contingency) Initial sync reactive outputs after contingency i;
parameter Q_wind_ck_0(i_wind, i_contingency) Initial wind reactive outputs after contingency i;
parameter Q_syncon_ck_0(i_syncon, i_contingency) Initial syncon reactive outputs after contingency i;
parameter Q_statcom_ck_0(i_statcom, i_contingency) Initial statcom reactive outputs after contingency i;
parameter Q_statvar_ck_0(i_statvar, i_contingency) Initial statvar reactive outputs after contingency i;
parameter Q_hvdc_embedded_1_ck_0(i_hvdc_embedded_Q, i_contingency) Initial hvdc_embedded_1 reactive outputs after contingency i;
parameter Q_hvdc_embedded_2_ck_0(i_hvdc_embedded_Q, i_contingency) Initial hvdc_embedded_2 reactive outputs after contingency i;
parameter Q_hvdc_interconnection_ck_0(i_hvdc_interconnection, i_contingency) Initial hvdc_interconnection reactive outputs after contingency i;
parameter Q_hvdc_spit_ck_0(i_hvdc_spit, i_contingency) Initial hvdc_spit reactive outputs after contingency i;

parameter droop(i_sync) droop of synchronous generators;

*BUS DATA

parameter demand(i_bus) active load at bus s;
parameter demandQ(i_bus) reactive load at bus s;

parameter V_0(i_bus);
parameter theta_0(i_bus);
parameter V_ck_0(i_bus, i_contingency);
parameter theta_ck_0(i_bus, i_contingency);


*LINES DATA

parameter branch_map(i_branch,i_bus) line map;
parameter B4_map(i_branch, i_B4) branches map;
parameter B6_map(i_branch, i_B6) branches map;
parameter Gff(i_branch) line conductances (from-from);
parameter Gft(i_branch) line conductances (from-to);
parameter Gtf(i_branch) line conductances (to-from);
parameter Gtt(i_branch) line conductances (to-to);
parameter Bff(i_branch) line susceptances (from-from);
parameter Bft(i_branch) line susceptances (from-to);
parameter Btf(i_branch) line susceptances (to-from);
parameter Btt(i_branch) line susceptances (to-to);

parameter branch_max_N(i_branch) continuous line ratings;
parameter branch_max_E(i_branch) emergency line ratings;

parameter contingency_states(i_branch, i_contingency) line contingencies;

parameter P1_0(i_branch);
parameter Q1_0(i_branch);
parameter P2_0(i_branch);
parameter Q2_0(i_branch);

parameter B4_flow_0 Initial total flow through B4 boundary;
parameter B6_flow_0 Initial total flow through B6 boundary;
parameter B4_flow_max Max total flow through B4 boundary;
parameter B6_flow_max Max total flow through B6 boundary;

parameter P1_ck_0(i_branch, i_contingency);
parameter Q1_ck_0(i_branch, i_contingency);
parameter P2_ck_0(i_branch, i_contingency);
parameter Q2_ck_0(i_branch, i_contingency);

parameter Vdev_pos_ck_0(i_bus, i_contingency) initial value for voltage deviations for contingency of line i;
parameter Vdev_neg_ck_0(i_bus, i_contingency) initial value for voltage deviations for contingency of line i;

$gdxin PrePSCACOPF
$load i_sync i_wind i_syncon i_statcom i_shunt i_statvar i_bus i_branch i_hvdc_embedded i_hvdc_embedded_Q i_hvdc_interconnection i_hvdc_spit i_dispatchable_load i_contingency i_B4 i_B6 droop sync_map wind_map syncon_map statcom_map shunt_map statvar_map hvdc_embedded_map hvdc_embedded_map_Q1 hvdc_embedded_map_Q2 hvdc_interconnection_map hvdc_spit_map dispatchable_load_map lincost hvdc_interconnection_costs dispatchable_load_costs sync_min sync_max sync_Qmin sync_Qmax wind_max wind_Qmin wind_Qmax syncon_Qmin syncon_Qmax statcom_Qmin statcom_Qmax shunt_Qmin shunt_Qmax statvar_Qmin statvar_Qmax hvdc_embedded_min hvdc_embedded_max hvdc_embedded_Qmin hvdc_embedded_Qmax hvdc_interconnection_min hvdc_interconnection_max hvdc_interconnection_Qmin hvdc_interconnection_Qmax hvdc_spit_min hvdc_spit_max hvdc_spit_total_max hvdc_spit_Qmin hvdc_spit_Qmax dispatchable_load_min dispatchable_load_max P_sync_0 P_wind_0 P_hvdc_embedded_0 P_hvdc_interconnection_0 P_hvdc_spit_0 P_dispatchable_load_0 Q_sync_0 Q_wind_0 Q_syncon_0 Q_statcom_0 Q_shunt_0 Q_statvar_0 Q_hvdc_embedded_1_0 Q_hvdc_embedded_2_0 Q_hvdc_interconnection_0 Q_hvdc_spit_0 Q_sync_ck_0 Q_wind_ck_0 Q_syncon_ck_0 Q_statcom_ck_0 Q_statvar_ck_0 Q_hvdc_embedded_1_ck_0 Q_hvdc_embedded_2_ck_0 Q_hvdc_interconnection_ck_0 Q_hvdc_spit_ck_0 demand demandQ V_0 theta_0 V_ck_0 theta_ck_0 branch_map Gff Gft Gtf Gtt Bff Bft Btf Btt branch_max_N branch_max_E contingency_states P1_0 Q1_0 P2_0 Q2_0 P1_ck_0 Q1_ck_0 P2_ck_0 Q2_ck_0 Vdev_pos_ck_0 Vdev_neg_ck_0 B4_flow_0 B6_flow_0 B4_flow_max B6_flow_max B4_map B6_map
$gdxin

***************************************************************
*** VARIABLES
***************************************************************

variable deviation dev
variable cost generator costs
variable Q_penalty penalty to push reactive setpoints towards the center of allowable range

positive variable P_sync(i_sync) sync generator outputs
positive variable P_wind(i_wind) wind generator outputs
variable P_hvdc_embedded(i_hvdc_embedded) power setpoint of embedded hvdc links
variable P_hvdc_interconnection(i_hvdc_interconnection) power setpoint of hvdc interconnections
variable P_hvdc_spit(i_hvdc_spit) power setpoint of spit hvdcs
variable P_dispatchable_load(i_dispatchable_load) power setpoint of dispatchable loads

variable Q_sync(i_sync) reactive sync generator outputs
variable Q_wind(i_wind) reactive wind outputs
variable Q_syncon(i_syncon) reactive syncon outputs
variable Q_statcom(i_statcom) reactive statcom outputs
variable Q_shunt(i_shunt) reactive shunt outputs
variable Q_statvar(i_statvar) reactive statvar outputs
variable Q_hvdc_embedded_1(i_hvdc_embedded_Q) reactive power setpoint of embedded hvdc links side 1
variable Q_hvdc_embedded_2(i_hvdc_embedded_Q) reactive power setpoint of embedded hvdc links side 2
variable Q_hvdc_interconnection(i_hvdc_interconnection) reactive power setpoint of hvdc interconnections
variable Q_hvdc_spit(i_hvdc_spit) reactive power setpoint of spit hvdcs

positive variable P_sync_ck(i_sync, i_contingency) sync generator outputs for contingency of line i
positive variable P_wind_ck(i_wind, i_contingency) wind generator outputs for contingency of line i
variable P_hvdc_embedded_ck(i_hvdc_embedded, i_contingency) power setpoint of embedded hvdc links for contingency of line i
variable P_hvdc_interconnection_ck(i_hvdc_interconnection, i_contingency) power setpoint of hvdc interconnections for contingency of line i
variable P_hvdc_spit_ck(i_hvdc_spit, i_contingency) power setpoint of hvdc spits for contingency of line i
variable P_dispatchable_load_ck(i_dispatchable_load, i_contingency) power setpoint of dispatchable loads for contingency of line i

variable Q_sync_ck(i_sync, i_contingency) reactive sync generator outputs for contingency of line i
variable Q_wind_ck(i_wind, i_contingency) reactive wind outputs for contingency of line i
variable Q_syncon_ck(i_syncon, i_contingency) reactive syncon outputs for contingency of line i
variable Q_statcom_ck(i_statcom, i_contingency) reactive statcom outputs for contingency of line i
variable Q_statvar_ck(i_statvar, i_contingency) reactive statvar outputs for contingency of line i
variable Q_hvdc_embedded_1_ck(i_hvdc_embedded_Q, i_contingency) reactive power setpoint of embedded hvdc links side 1 for contingency of line i
variable Q_hvdc_embedded_2_ck(i_hvdc_embedded_Q, i_contingency) reactive power setpoint of embedded hvdc links side 2 for contingency of line i
variable Q_hvdc_interconnection_ck(i_hvdc_interconnection, i_contingency) reactive power setpoint of hvdc interconnections for contingency of line i
variable Q_hvdc_spit_ck(i_hvdc_spit, i_contingency) reactive power setpoint of hvdc spits for contingency of line i

positive variable V(i_bus) bus voltage amplitude in base state
positive variable V_ck(i_bus, i_contingency) bus voltage amplitude in line contingency state
variable theta(i_bus) bus voltage angles in base state
variable theta_ck(i_bus, i_contingency) bus voltage angles in line contingency state

positive variable Vdev_pos_ck(i_bus, i_contingency) positive voltage deviation after contingency i;
positive variable Vdev_neg_ck(i_bus, i_contingency) negative voltage deviation after contingency i;

variable P1(i_branch) active power flow through lines in base state
variable Q1(i_branch) reactive power flow through lines in base state
variable P2(i_branch) active power flow through lines in base state
variable Q2(i_branch) reactive power flow through lines in base state

variable B4_flow total flow through boundary B4
variable B6_flow total flow through boundary B6

variable P1_ck(i_branch, i_contingency) active power flow through lines in line contingency state
variable Q1_ck(i_branch, i_contingency) reactive power flow through lines in line contingency state
variable P2_ck(i_branch, i_contingency) active power flow through lines in line contingency state
variable Q2_ck(i_branch, i_contingency) reactive power flow through lines in line contingency state

variable DeltaF_ck(i_contingency) frequency deviation after line contingency

***************************************************************
*** EQUATION DECLARATION
***************************************************************

equations

dev objective
cost_eq generator costs
Q_penalty_eq penalty to push reactive setpoints towards the center of allowable range
Pg_sync_min(i_sync) minimum generator active output
Pg_sync_max(i_sync) maximum generator active output
Pg_wind_max(i_wind)  maximum wind generator active output
Pg_hvdc_embedded_min(i_hvdc_embedded) minimum embedded hvdc active output
Pg_hvdc_embedded_max(i_hvdc_embedded) maximum embedded hvdc active output
Pg_hvdc_interconnection_min(i_hvdc_interconnection) minimum hvdc interconection active output
Pg_hvdc_interconnection_max(i_hvdc_interconnection) maximum hvdc interconection active output
Pg_hvdc_spit_min(i_hvdc_spit) minimum spit hvdc active output
Pg_hvdc_spit_max(i_hvdc_spit) maximum spit hvdc active output
Pg_hvdc_spit_total_max maximum available wind for spit hvdcs
Pg_dispatchable_load_min(i_dispatchable_load) minimum dispatchable load active output
Pg_dispatchable_load_max(i_dispatchable_load) maximum dispatchable load active output
Qg_sync_min(i_sync) minimum generator reactive output
Qg_sync_max(i_sync) maximum generator reactive output
Qg_wind_min(i_wind)  inximum wind generator reactive output
Qg_wind_max(i_wind)  maximum wind generator reactive output
Qg_syncon_min(i_syncon) minimum syncon reactive output
Qg_syncon_max(i_syncon) maximum syncon reactive output
Qg_statcom_min(i_statcom) minimum statcom reactive output
Qg_statcom_max(i_statcom) maximum statcom reactive output
Qg_shunt_min(i_shunt) minimum shunt reactive output
Qg_shunt_max(i_shunt) maximum shunt reactive output
Qg_statvar_min(i_statvar) minimum statvar reactive output
Qg_statvar_max(i_statvar) maximum statvar reactive output
Qg_hvdc_embedded_min_1(i_hvdc_embedded_Q) minimum embedded hvdc reactive output side 1
Qg_hvdc_embedded_max_1(i_hvdc_embedded_Q) maximum embedded hvdc reactive output side 1
Qg_hvdc_embedded_min_2(i_hvdc_embedded_Q) minimum embedded hvdc reactive output side 2
Qg_hvdc_embedded_max_2(i_hvdc_embedded_Q) maximum embedded hvdc reactive output side 2
Qg_hvdc_interconnection_min(i_hvdc_interconnection) minimum hvdc interconection reactive output
Qg_hvdc_interconnection_max(i_hvdc_interconnection) maximum hvdc interconection reactive output
Qg_hvdc_spit_min(i_hvdc_spit) minimum hvdc interconection reactive output
Qg_hvdc_spit_max(i_hvdc_spit) maximum hvdc interconection reactive output
P_balance(i_bus) active power balance for each bus
Q_balance(i_bus) active power balance for each bus
Voltage_min(i_bus) voltage minimum limit
Voltage_max(i_bus) voltage maximum limit
Angles_min(i_bus) voltage angles negative limit
Angles_max(i_bus) voltage angles positive limit
line_P1(i_branch) defining power flow through lines
line_Q1(i_branch) defining power flow through lines
line_P2(i_branch) defining power flow through lines
line_Q2(i_branch) defining power flow through lines
line_max1(i_branch) continuous line rating
line_max2(i_branch) continuous line rating
boundary_B4 defining total power flow through boundary B4
boundary_B6 defining total power flow through boundary B6
boundary_B4_max maximum flow through boundary B4
boundary_B6_max maximum flow through boundary B6
SPowerDevck(i_sync, i_contingency) power deviation after a line contingency
Vdev(i_bus, i_contingency) voltage deviation from generator setpoint
Vdev2(i_bus, i_contingency) voltage deviation from generator setpoint

Qg_sync_min_ck(i_sync, i_contingency) minimum generator active output
Qg_sync_max_ck(i_sync, i_contingency) maximum generator active output
sync_PQswitchMax(i_sync, i_contingency) PQ switch
sync_PQswitchMin(i_sync, i_contingency) PQ switch
Qg_wind_min_ck(i_wind, i_contingency) minimum generator active output
Qg_wind_max_ck(i_wind, i_contingency) maximum generator active output
wind_PQswitchMax(i_wind, i_contingency) PQ switch
wind_PQswitchMin(i_wind, i_contingency) PQ switch
Qg_syncon_min_ck(i_syncon, i_contingency) minimum generator active output
Qg_syncon_max_ck(i_syncon, i_contingency) maximum generator active output
syncon_PQswitchMax(i_syncon, i_contingency) PQ switch
syncon_PQswitchMin(i_syncon, i_contingency) PQ switch
Qg_statcom_min_ck(i_statcom, i_contingency) minimum generator active output
Qg_statcom_max_ck(i_statcom, i_contingency) maximum generator active output
statcom_PQswitchMax(i_statcom, i_contingency) PQ switch
statcom_PQswitchMin(i_statcom, i_contingency) PQ switch
Qg_statvar_min_ck(i_statvar, i_contingency) minimum generator active output
Qg_statvar_max_ck(i_statvar, i_contingency) maximum generator active output
statvar_PQswitchMax(i_statvar, i_contingency) PQ switch
statvar_PQswitchMin(i_statvar, i_contingency) PQ switch
Qg_hvdc_embedded_1_min_ck(i_hvdc_embedded_Q, i_contingency) minimum generator active output
Qg_hvdc_embedded_1_max_ck(i_hvdc_embedded_Q, i_contingency) maximum generator active output
hvdc_embedded_1_PQswitchMax(i_hvdc_embedded_Q, i_contingency) PQ switch
hvdc_embedded_1_PQswitchMin(i_hvdc_embedded_Q, i_contingency) PQ switch
Qg_hvdc_embedded_2_min_ck(i_hvdc_embedded_Q, i_contingency) minimum generator active output
Qg_hvdc_embedded_2_max_ck(i_hvdc_embedded_Q, i_contingency) maximum generator active output
hvdc_embedded_2_PQswitchMax(i_hvdc_embedded_Q, i_contingency) PQ switch
hvdc_embedded_2_PQswitchMin(i_hvdc_embedded_Q, i_contingency) PQ switch
Qg_hvdc_interconnection_min_ck(i_hvdc_interconnection, i_contingency) minimum generator active output
Qg_hvdc_interconnection_max_ck(i_hvdc_interconnection, i_contingency) maximum generator active output
hvdc_interconnection_PQswitchMax(i_hvdc_interconnection, i_contingency) PQ switch
hvdc_interconnection_PQswitchMin(i_hvdc_interconnection, i_contingency) PQ switch
Qg_hvdc_spit_min_ck(i_hvdc_spit, i_contingency) minimum generator active output
Qg_hvdc_spit_max_ck(i_hvdc_spit, i_contingency) maximum generator active output
hvdc_spit_PQswitchMax(i_hvdc_spit, i_contingency) PQ switch
hvdc_spit_PQswitchMin(i_hvdc_spit, i_contingency) PQ switch

Voltage_min_ck(i_bus, i_contingency) voltage minimum limit
Voltage_max_ck(i_bus, i_contingency) voltage maximum limit
Angles_min_ck(i_bus, i_contingency) voltage angles negative limit
Angles_max_ck(i_bus, i_contingency) voltage angles positive limit
P_balance_ck(i_bus, i_contingency) active power balance for each bus
Q_balance_ck(i_bus, i_contingency) active power balance for each bus
line_P1_ck(i_branch, i_contingency) defining power flow through lines
line_Q1_ck(i_branch, i_contingency) defining power flow through lines
line_P2_ck(i_branch, i_contingency) defining power flow through lines
line_Q2_ck(i_branch, i_contingency) defining power flow through lines
line_max1_ck(i_branch, i_contingency) emergency line rating
line_max2_ck(i_branch, i_contingency) emergency line rating
;


***************************************************************
*** SETTINGS
***************************************************************

P_sync.l(i_sync) = P_sync_0(i_sync);
P_wind.l(i_wind) = P_wind_0(i_wind);
P_hvdc_embedded.l(i_hvdc_embedded) = P_hvdc_embedded_0(i_hvdc_embedded);
P_hvdc_interconnection.l(i_hvdc_interconnection) = P_hvdc_interconnection_0(i_hvdc_interconnection);
P_hvdc_spit.l(i_hvdc_spit) = P_hvdc_spit_0(i_hvdc_spit);
P_dispatchable_load.l(i_dispatchable_load) = P_dispatchable_load_0(i_dispatchable_load);

Q_sync.l(i_sync) = Q_sync_0(i_sync);
Q_wind.l(i_wind) = Q_wind_0(i_wind);
Q_syncon.l(i_syncon) = Q_syncon_0(i_syncon);
Q_statcom.l(i_statcom) = Q_statcom_0(i_statcom);
Q_shunt.l(i_shunt) = Q_shunt_0(i_shunt);
Q_statvar.l(i_statvar) = Q_statvar_0(i_statvar);
Q_hvdc_embedded_1.l(i_hvdc_embedded_Q) = Q_hvdc_embedded_1_0(i_hvdc_embedded_Q);
Q_hvdc_embedded_2.l(i_hvdc_embedded_Q) = Q_hvdc_embedded_2_0(i_hvdc_embedded_Q);
Q_hvdc_interconnection.l(i_hvdc_interconnection) = Q_hvdc_interconnection_0(i_hvdc_interconnection);
Q_hvdc_spit.l(i_hvdc_spit) = Q_hvdc_spit_0(i_hvdc_spit);

P_sync_ck.l(i_sync, i_contingency) = P_sync_0(i_sync);
P_wind_ck.l(i_wind, i_contingency) = P_wind_0(i_wind);
P_hvdc_embedded_ck.l(i_hvdc_embedded, i_contingency) = P_hvdc_embedded_0(i_hvdc_embedded);
P_hvdc_interconnection_ck.l(i_hvdc_interconnection, i_contingency) = P_hvdc_interconnection_0(i_hvdc_interconnection);
P_hvdc_spit_ck.l(i_hvdc_spit, i_contingency) = P_hvdc_spit_0(i_hvdc_spit);
P_dispatchable_load_ck.l(i_dispatchable_load, i_contingency) = P_dispatchable_load_0(i_dispatchable_load);

Q_sync_ck.l(i_sync, i_contingency) = Q_sync_ck_0(i_sync, i_contingency);
Q_wind_ck.l(i_wind, i_contingency) = Q_wind_ck_0(i_wind, i_contingency);
Q_syncon_ck.l(i_syncon, i_contingency) = Q_syncon_ck_0(i_syncon, i_contingency);
Q_statcom_ck.l(i_statcom, i_contingency) = Q_statcom_ck_0(i_statcom, i_contingency);
Q_statvar_ck.l(i_statvar, i_contingency) = Q_statvar_ck_0(i_statvar, i_contingency);
Q_hvdc_embedded_1_ck.l(i_hvdc_embedded_Q, i_contingency) = Q_hvdc_embedded_1_ck_0(i_hvdc_embedded_Q, i_contingency);
Q_hvdc_embedded_2_ck.l(i_hvdc_embedded_Q, i_contingency) = Q_hvdc_embedded_2_ck_0(i_hvdc_embedded_Q, i_contingency);
Q_hvdc_interconnection_ck.l(i_hvdc_interconnection, i_contingency) = Q_hvdc_interconnection_ck_0(i_hvdc_interconnection, i_contingency);
Q_hvdc_spit_ck.l(i_hvdc_spit, i_contingency) = Q_hvdc_spit_ck_0(i_hvdc_spit, i_contingency);

V.l(i_bus) = V_0(i_bus);
V_ck.l(i_bus, i_contingency) = V_ck_0(i_bus, i_contingency);
Vdev_pos_ck.l(i_bus, i_contingency) = Vdev_pos_ck_0(i_bus, i_contingency);
Vdev_neg_ck.l(i_bus, i_contingency) = Vdev_neg_ck_0(i_bus, i_contingency);
theta.l(i_bus) = theta_0(i_bus);
theta_ck.l(i_bus, i_contingency) = theta_ck_0(i_bus, i_contingency);

P1.l(i_branch) = P1_0(i_branch);
Q1.l(i_branch) = Q1_0(i_branch);
P2.l(i_branch) = P2_0(i_branch);
Q2.l(i_branch) = Q2_0(i_branch);

B4_flow.l = B4_flow_0;
B6_flow.l = B6_flow_0;

P1_ck.l(i_branch, i_contingency) = P1_ck_0(i_branch, i_contingency);
Q1_ck.l(i_branch, i_contingency) = Q1_ck_0(i_branch, i_contingency);
P2_ck.l(i_branch, i_contingency) = P2_ck_0(i_branch, i_contingency);
Q2_ck.l(i_branch, i_contingency) = Q2_ck_0(i_branch, i_contingency);


***************************************************************
*** EQUATIONS
***************************************************************

dev..
deviation =e=
sum(i_sync, power(P_sync(i_sync) - P_sync_0(i_sync), 2))
+ sum(i_wind, power(P_wind(i_wind) - P_wind_0(i_wind), 2))
+ sum(i_hvdc_embedded, power(P_hvdc_embedded(i_hvdc_embedded) - P_hvdc_embedded_0(i_hvdc_embedded), 2))
+ sum(i_hvdc_interconnection, power(P_hvdc_interconnection(i_hvdc_interconnection) - P_hvdc_interconnection_0(i_hvdc_interconnection), 2))
+ sum(i_hvdc_spit, power(P_hvdc_spit(i_hvdc_spit) - P_hvdc_spit_0(i_hvdc_spit), 2))
+ sum(i_dispatchable_load, power(P_dispatchable_load(i_dispatchable_load) - P_dispatchable_load_0(i_dispatchable_load), 2))
+ Kg * sum(i_sync, power(Q_sync(i_sync) - Q_sync_0(i_sync), 2))
+ Kg * sum(i_wind, power(Q_wind(i_wind) - Q_wind_0(i_wind), 2))
+ Kg * sum(i_syncon, power(Q_syncon(i_syncon) - Q_syncon_0(i_syncon), 2))
+ Kg * sum(i_statcom, power(Q_statcom(i_statcom) - Q_statcom_0(i_statcom), 2))
+ Kg * sum(i_statvar, power(Q_statvar(i_statvar) - Q_statvar_0(i_statvar), 2))
+ Kg * sum(i_hvdc_embedded_Q, power(Q_hvdc_embedded_1(i_hvdc_embedded_Q) - Q_hvdc_embedded_1_0(i_hvdc_embedded_Q), 2))
+ Kg * sum(i_hvdc_embedded_Q, power(Q_hvdc_embedded_2(i_hvdc_embedded_Q) - Q_hvdc_embedded_2_0(i_hvdc_embedded_Q), 2))
+ Kg * sum(i_hvdc_interconnection, power(Q_hvdc_interconnection(i_hvdc_interconnection) - Q_hvdc_interconnection_0(i_hvdc_interconnection), 2))
+ Kg * sum(i_hvdc_spit, power(Q_hvdc_spit(i_hvdc_spit) - Q_hvdc_spit_0(i_hvdc_spit), 2))
+ 0 * Q_penalty;
* Q_penalty directly written in objective such that the solver cannot change it

* Need to initialise the cost, otherwise optimisation diverges
cost_eq..
cost =e= 0;
*sum(i_sync, P_sync(i_sync) * lincost(i_sync))
*       + sum(i_hvdc_interconnection, P_hvdc_interconnection(i_hvdc_interconnection) * hvdc_interconnection_costs(i_hvdc_interconnection))
*       - sum(i_dispatchable_load, P_dispatchable_load(i_dispatchable_load) * dispatchable_load_costs(i_dispatchable_load));

Q_penalty_eq..
Q_penalty =e= sum(i_sync, power(Q_sync(i_sync) - (sync_Qmax(i_sync) + sync_Qmin(i_sync))/2, 2) / power(sync_Qmax(i_sync) - sync_Qmin(i_sync) + Epsilon, 2))
            + sum(i_wind, power(Q_wind(i_wind) - (wind_Qmax(i_wind) + wind_Qmin(i_wind))/2, 2) / power(wind_Qmax(i_wind) - wind_Qmin(i_wind) + Epsilon, 2))
            + sum(i_syncon, power(Q_syncon(i_syncon) - (syncon_Qmax(i_syncon) + syncon_Qmin(i_syncon))/2, 2) / power(syncon_Qmax(i_syncon) - syncon_Qmin(i_syncon) + Epsilon, 2))
            + sum(i_statcom, power(Q_statcom(i_statcom) - (statcom_Qmax(i_statcom) + statcom_Qmin(i_statcom))/2, 2) / power(statcom_Qmax(i_statcom) - statcom_Qmin(i_statcom) + Epsilon, 2))
            + sum(i_statvar, power(Q_statvar(i_statvar) - (statvar_Qmax(i_statvar) + statvar_Qmin(i_statvar))/2, 2) / power(statvar_Qmax(i_statvar) - statvar_Qmin(i_statvar) + Epsilon, 2))
            + sum(i_hvdc_embedded_Q, power(Q_hvdc_embedded_1(i_hvdc_embedded_Q) - (hvdc_embedded_Qmax(i_hvdc_embedded_Q) + hvdc_embedded_Qmin(i_hvdc_embedded_Q))/2, 2) / power(hvdc_embedded_Qmax(i_hvdc_embedded_Q) - hvdc_embedded_Qmin(i_hvdc_embedded_Q) + Epsilon, 2))
            + sum(i_hvdc_embedded_Q, power(Q_hvdc_embedded_2(i_hvdc_embedded_Q) - (hvdc_embedded_Qmax(i_hvdc_embedded_Q) + hvdc_embedded_Qmin(i_hvdc_embedded_Q))/2, 2) / power(hvdc_embedded_Qmax(i_hvdc_embedded_Q) - hvdc_embedded_Qmin(i_hvdc_embedded_Q) + Epsilon, 2))
            + sum(i_hvdc_interconnection, power(Q_hvdc_interconnection(i_hvdc_interconnection) - (hvdc_interconnection_Qmax(i_hvdc_interconnection) + hvdc_interconnection_Qmin(i_hvdc_interconnection))/2, 2) / power(hvdc_interconnection_Qmax(i_hvdc_interconnection) - hvdc_interconnection_Qmin(i_hvdc_interconnection) + Epsilon, 2))
            + sum(i_hvdc_spit, power(Q_hvdc_spit(i_hvdc_spit) - (hvdc_spit_Qmax(i_hvdc_spit) + hvdc_spit_Qmin(i_hvdc_spit))/2, 2) / power(hvdc_spit_Qmax(i_hvdc_spit) - hvdc_spit_Qmin(i_hvdc_spit) + Epsilon, 2));
* Mechanically-switched shunts are not included in the Q_penalty has they do not provide dynamic support

Pg_sync_min(i_sync)..
P_sync(i_sync) =g= sync_min(i_sync);

Pg_sync_max(i_sync)..
P_sync(i_sync) =l= sync_max(i_sync);

Pg_wind_max(i_wind)..
P_wind(i_wind) =l= wind_max(i_wind);

Pg_hvdc_embedded_min(i_hvdc_embedded)..
P_hvdc_embedded(i_hvdc_embedded) =g= hvdc_embedded_min(i_hvdc_embedded);

Pg_hvdc_embedded_max(i_hvdc_embedded)..
P_hvdc_embedded(i_hvdc_embedded) =l= hvdc_embedded_max(i_hvdc_embedded);

Pg_hvdc_interconnection_min(i_hvdc_interconnection)..
P_hvdc_interconnection(i_hvdc_interconnection) =g= hvdc_interconnection_min(i_hvdc_interconnection);

Pg_hvdc_interconnection_max(i_hvdc_interconnection)..
P_hvdc_interconnection(i_hvdc_interconnection) =l= hvdc_interconnection_max(i_hvdc_interconnection);

Pg_hvdc_spit_min(i_hvdc_spit)..
P_hvdc_spit(i_hvdc_spit) =g= hvdc_spit_min(i_hvdc_spit);

Pg_hvdc_spit_max(i_hvdc_spit)..
P_hvdc_spit(i_hvdc_spit) =l= hvdc_spit_max(i_hvdc_spit);

Pg_hvdc_spit_total_max..
sum(i_hvdc_spit, P_hvdc_spit(i_hvdc_spit)) =l= hvdc_spit_total_max;

Pg_dispatchable_load_min(i_dispatchable_load)..
P_dispatchable_load(i_dispatchable_load) =g= dispatchable_load_min(i_dispatchable_load);

Pg_dispatchable_load_max(i_dispatchable_load)..
P_dispatchable_load(i_dispatchable_load) =l= dispatchable_load_max(i_dispatchable_load);

Qg_sync_min(i_sync)..
Q_sync(i_sync) =g= sync_Qmin(i_sync);

Qg_sync_max(i_sync)..
Q_sync(i_sync) =l= sync_Qmax(i_sync);

Qg_wind_min(i_wind)..
Q_wind(i_wind) =g= wind_Qmin(i_wind);

Qg_wind_max(i_wind)..
Q_wind(i_wind) =l= wind_Qmax(i_wind);

Qg_syncon_min(i_syncon)..
Q_syncon(i_syncon) =g= syncon_Qmin(i_syncon);

Qg_syncon_max(i_syncon)..
Q_syncon(i_syncon) =l= syncon_Qmax(i_syncon);

Qg_statcom_min(i_statcom)..
Q_statcom(i_statcom) =g= statcom_Qmin(i_statcom);

Qg_statcom_max(i_statcom)..
Q_statcom(i_statcom) =l= statcom_Qmax(i_statcom);

Qg_shunt_min(i_shunt)..
Q_shunt(i_shunt) =g= shunt_Qmin(i_shunt);

Qg_shunt_max(i_shunt)..
Q_shunt(i_shunt) =l= shunt_Qmax(i_shunt);

Qg_statvar_min(i_statvar)..
Q_statvar(i_statvar) =g= statvar_Qmin(i_statvar);

Qg_statvar_max(i_statvar)..
Q_statvar(i_statvar) =l= statvar_Qmax(i_statvar);

Qg_hvdc_embedded_min_1(i_hvdc_embedded_Q)..
Q_hvdc_embedded_1(i_hvdc_embedded_Q) =g= hvdc_embedded_Qmin(i_hvdc_embedded_Q);

Qg_hvdc_embedded_max_1(i_hvdc_embedded_Q)..
Q_hvdc_embedded_1(i_hvdc_embedded_Q) =l= hvdc_embedded_Qmax(i_hvdc_embedded_Q);

Qg_hvdc_embedded_min_2(i_hvdc_embedded_Q)..
Q_hvdc_embedded_2(i_hvdc_embedded_Q) =g= hvdc_embedded_Qmin(i_hvdc_embedded_Q);

Qg_hvdc_embedded_max_2(i_hvdc_embedded_Q)..
Q_hvdc_embedded_2(i_hvdc_embedded_Q) =l= hvdc_embedded_Qmax(i_hvdc_embedded_Q);

Qg_hvdc_interconnection_min(i_hvdc_interconnection)..
Q_hvdc_interconnection(i_hvdc_interconnection) =g= hvdc_interconnection_Qmin(i_hvdc_interconnection);

Qg_hvdc_interconnection_max(i_hvdc_interconnection)..
Q_hvdc_interconnection(i_hvdc_interconnection) =l= hvdc_interconnection_Qmax(i_hvdc_interconnection);

Qg_hvdc_spit_min(i_hvdc_spit)..
Q_hvdc_spit(i_hvdc_spit) =g= hvdc_spit_Qmin(i_hvdc_spit);

Qg_hvdc_spit_max(i_hvdc_spit)..
Q_hvdc_spit(i_hvdc_spit) =l= hvdc_spit_Qmax(i_hvdc_spit);

P_balance(i_bus)..
sum(i_sync$(sync_map(i_sync, i_bus)), P_sync(i_sync))
+ sum(i_wind$(wind_map(i_wind, i_bus)), P_wind(i_wind))
+ sum(i_hvdc_embedded, P_hvdc_embedded(i_hvdc_embedded)*hvdc_embedded_map(i_hvdc_embedded, i_bus))
+ sum(i_hvdc_interconnection$(hvdc_interconnection_map(i_hvdc_interconnection, i_bus)), P_hvdc_interconnection(i_hvdc_interconnection))
+ sum(i_hvdc_spit$(hvdc_spit_map(i_hvdc_spit, i_bus)), P_hvdc_spit(i_hvdc_spit))
- sum(i_dispatchable_load$(dispatchable_load_map(i_dispatchable_load, i_bus)), P_dispatchable_load(i_dispatchable_load))
- demand(i_bus)
=e=
sum(i_branch$(branch_map(i_branch,i_bus) = 1),P1(i_branch))+sum(i_branch$(branch_map(i_branch,i_bus) = -1),P2(i_branch));

* Statvars follow a receptor convention in powerfactory (shunts too but accounted in Python code)
Q_balance(i_bus)..
sum(i_sync$(sync_map(i_sync, i_bus)), Q_sync(i_sync))
+ sum(i_wind$(wind_map(i_wind, i_bus)), Q_wind(i_wind))
+ sum(i_syncon$(syncon_map(i_syncon, i_bus)), Q_syncon(i_syncon))
+ sum(i_statcom$(statcom_map(i_statcom, i_bus)), Q_statcom(i_statcom))
+ sum(i_shunt$(shunt_map(i_shunt, i_bus)), Q_shunt(i_shunt)*V(i_bus)*V(i_bus))
- sum(i_statvar$(statvar_map(i_statvar, i_bus)), Q_statvar(i_statvar))
+ sum(i_hvdc_embedded_Q, Q_hvdc_embedded_1(i_hvdc_embedded_Q)*hvdc_embedded_map_Q1(i_hvdc_embedded_Q, i_bus))
+ sum(i_hvdc_embedded_Q, Q_hvdc_embedded_2(i_hvdc_embedded_Q)*hvdc_embedded_map_Q2(i_hvdc_embedded_Q, i_bus))
+ sum(i_hvdc_interconnection$(hvdc_interconnection_map(i_hvdc_interconnection, i_bus)), Q_hvdc_interconnection(i_hvdc_interconnection))
+ sum(i_hvdc_spit$(hvdc_spit_map(i_hvdc_spit, i_bus)), Q_hvdc_spit(i_hvdc_spit))
- demandQ(i_bus)
=e=
sum(i_branch$(branch_map(i_branch,i_bus) = 1),Q1(i_branch))+sum(i_branch$(branch_map(i_branch,i_bus) = -1),Q2(i_branch));

Voltage_min(i_bus)..
V(i_bus) =g= 0.98;

Voltage_max(i_bus)..
V(i_bus) =l= 1.05;

Angles_min(i_bus)..
theta(i_bus) =g= -pi;

Angles_max(i_bus)..
theta(i_bus) =l= pi;

line_P1(i_branch)..
P1(i_branch)
=e=
sum(i_bus$(branch_map(i_branch,i_bus) = 1),(sum(j_bus$(branch_map(i_branch,j_bus)=-1), V(i_bus)*(V(i_bus)*Gff(i_branch)+V(j_bus)*Gft(i_branch)*cos(theta(i_bus)-theta(j_bus))+V(j_bus)*Bft(i_branch)*sin(theta(i_bus)-theta(j_bus))))));

line_Q1(i_branch)..
Q1(i_branch)
=e=
sum(i_bus$(branch_map(i_branch,i_bus) = 1),(sum(j_bus$(branch_map(i_branch,j_bus)=-1), V(i_bus)*(-V(i_bus)*Bff(i_branch)+V(j_bus)*Gft(i_branch)*sin(theta(i_bus)-theta(j_bus))-V(j_bus)*Bft(i_branch)*cos(theta(i_bus)-theta(j_bus))))));

line_P2(i_branch)..
P2(i_branch)
=e=
sum(i_bus$(branch_map(i_branch,i_bus) = -1),(sum(j_bus$(branch_map(i_branch,j_bus)=1), V(i_bus)*(V(i_bus)*Gtt(i_branch)+V(j_bus)*Gtf(i_branch)*cos(theta(i_bus)-theta(j_bus))+V(j_bus)*Btf(i_branch)*sin(theta(i_bus)-theta(j_bus))))));

line_Q2(i_branch)..
Q2(i_branch)
=e=
sum(i_bus$(branch_map(i_branch,i_bus) = -1),(sum(j_bus$(branch_map(i_branch,j_bus)=1), V(i_bus)*(-V(i_bus)*Btt(i_branch)+V(j_bus)*Gtf(i_branch)*sin(theta(i_bus)-theta(j_bus))-V(j_bus)*Btf(i_branch)*cos(theta(i_bus)-theta(j_bus))))));

line_max1(i_branch)..
P1(i_branch)*P1(i_branch)+Q1(i_branch)*Q1(i_branch) =l= branch_max_N(i_branch)*branch_max_N(i_branch);

line_max2(i_branch)..
P2(i_branch)*P2(i_branch)+Q2(i_branch)*Q2(i_branch) =l= branch_max_N(i_branch)*branch_max_N(i_branch);

boundary_B4..
B4_flow =e= sum(i_branch, sum(i_B4, B4_map(i_branch, i_B4) * P1(i_branch) * P1(i_branch)));

boundary_B6..
B6_flow =e= sum(i_branch, sum(i_B6, B6_map(i_branch, i_B6) * P1(i_branch) * P1(i_branch)));

boundary_B4_max..
B4_flow =l= B4_flow_max * B4_flow_max;

boundary_B6_max..
B6_flow =l= B6_flow_max * B6_flow_max;

SPowerDevck(i_sync, i_contingency)..
P_sync_ck(i_sync, i_contingency) =l= P_sync(i_sync) - DeltaF_ck(i_contingency) * droop(i_sync);

Vdev(i_bus, i_contingency)..
V_ck(i_bus, i_contingency) - V(i_bus) - Vdev_pos_ck(i_bus, i_contingency) + Vdev_neg_ck(i_bus, i_contingency) =e= 0;

Vdev2(i_bus, i_contingency)..
Vdev_pos_ck(i_bus, i_contingency) * Vdev_neg_ck(i_bus, i_contingency) =l= 0;

Qg_sync_min_ck(i_sync, i_contingency)..
Q_sync_ck(i_sync, i_contingency) =g= sync_Qmin(i_sync);

Qg_sync_max_ck(i_sync, i_contingency)..
Q_sync_ck(i_sync, i_contingency) =l= sync_Qmax(i_sync);

sync_PQswitchMax(i_sync, i_contingency)..
(Q_sync_ck(i_sync, i_contingency) - sync_Qmax(i_sync)) * sum(i_bus$(sync_map(i_sync,i_bus)), Vdev_neg_ck(i_bus, i_contingency)) =e= 0;

sync_PQswitchMin(i_sync, i_contingency)..
(Q_sync_ck(i_sync, i_contingency) - sync_Qmin(i_sync)) * sum(i_bus$(sync_map(i_sync,i_bus)), Vdev_pos_ck(i_bus, i_contingency)) =e= 0;

Qg_wind_min_ck(i_wind, i_contingency)..
Q_wind_ck(i_wind, i_contingency) =g= wind_Qmin(i_wind);

Qg_wind_max_ck(i_wind, i_contingency)..
Q_wind_ck(i_wind, i_contingency) =l= wind_Qmax(i_wind);

wind_PQswitchMax(i_wind, i_contingency)..
(Q_wind_ck(i_wind, i_contingency) - wind_Qmax(i_wind)) * sum(i_bus$(wind_map(i_wind,i_bus)), Vdev_neg_ck(i_bus, i_contingency)) =e= 0;

wind_PQswitchMin(i_wind, i_contingency)..
(Q_wind_ck(i_wind, i_contingency) - wind_Qmin(i_wind)) * sum(i_bus$(wind_map(i_wind,i_bus)), Vdev_pos_ck(i_bus, i_contingency)) =e= 0;

Qg_syncon_min_ck(i_syncon, i_contingency)..
Q_syncon_ck(i_syncon, i_contingency) =g= syncon_Qmin(i_syncon);

Qg_syncon_max_ck(i_syncon, i_contingency)..
Q_syncon_ck(i_syncon, i_contingency) =l= syncon_Qmax(i_syncon);

syncon_PQswitchMax(i_syncon, i_contingency)..
(Q_syncon_ck(i_syncon, i_contingency) - syncon_Qmax(i_syncon)) * sum(i_bus$(syncon_map(i_syncon,i_bus)), Vdev_neg_ck(i_bus, i_contingency)) =e= 0;

syncon_PQswitchMin(i_syncon, i_contingency)..
(Q_syncon_ck(i_syncon, i_contingency) - syncon_Qmin(i_syncon)) * sum(i_bus$(syncon_map(i_syncon,i_bus)), Vdev_pos_ck(i_bus, i_contingency)) =e= 0;

Qg_statcom_min_ck(i_statcom, i_contingency)..
Q_statcom_ck(i_statcom, i_contingency) =g= statcom_Qmin(i_statcom);

Qg_statcom_max_ck(i_statcom, i_contingency)..
Q_statcom_ck(i_statcom, i_contingency) =l= statcom_Qmax(i_statcom);

statcom_PQswitchMax(i_statcom, i_contingency)..
(Q_statcom_ck(i_statcom, i_contingency) - statcom_Qmax(i_statcom)) * sum(i_bus$(statcom_map(i_statcom,i_bus)), Vdev_neg_ck(i_bus, i_contingency)) =e= 0;

statcom_PQswitchMin(i_statcom, i_contingency)..
(Q_statcom_ck(i_statcom, i_contingency) - statcom_Qmin(i_statcom)) * sum(i_bus$(statcom_map(i_statcom,i_bus)), Vdev_pos_ck(i_bus, i_contingency)) =e= 0;

Qg_statvar_min_ck(i_statvar, i_contingency)..
Q_statvar_ck(i_statvar, i_contingency) =g= statvar_Qmin(i_statvar);

Qg_statvar_max_ck(i_statvar, i_contingency)..
Q_statvar_ck(i_statvar, i_contingency) =l= statvar_Qmax(i_statvar);

statvar_PQswitchMax(i_statvar, i_contingency)..
(Q_statvar_ck(i_statvar, i_contingency) - statvar_Qmax(i_statvar)) * sum(i_bus$(statvar_map(i_statvar,i_bus)), Vdev_neg_ck(i_bus, i_contingency)) =e= 0;

statvar_PQswitchMin(i_statvar, i_contingency)..
(Q_statvar_ck(i_statvar, i_contingency) - statvar_Qmin(i_statvar)) * sum(i_bus$(statvar_map(i_statvar,i_bus)), Vdev_pos_ck(i_bus, i_contingency)) =e= 0;

Qg_hvdc_embedded_1_min_ck(i_hvdc_embedded_Q, i_contingency)..
Q_hvdc_embedded_1_ck(i_hvdc_embedded_Q, i_contingency) =g= hvdc_embedded_Qmin(i_hvdc_embedded_Q);

Qg_hvdc_embedded_1_max_ck(i_hvdc_embedded_Q, i_contingency)..
Q_hvdc_embedded_1_ck(i_hvdc_embedded_Q, i_contingency) =l= hvdc_embedded_Qmax(i_hvdc_embedded_Q);

hvdc_embedded_1_PQswitchMax(i_hvdc_embedded_Q, i_contingency)..
(Q_hvdc_embedded_1_ck(i_hvdc_embedded_Q, i_contingency) - hvdc_embedded_Qmax(i_hvdc_embedded_Q)) * sum(i_bus$(hvdc_embedded_map_Q1(i_hvdc_embedded_Q,i_bus)), Vdev_neg_ck(i_bus, i_contingency)) =e= 0;

hvdc_embedded_1_PQswitchMin(i_hvdc_embedded_Q, i_contingency)..
(Q_hvdc_embedded_1_ck(i_hvdc_embedded_Q, i_contingency) - hvdc_embedded_Qmin(i_hvdc_embedded_Q)) * sum(i_bus$(hvdc_embedded_map_Q1(i_hvdc_embedded_Q,i_bus)), Vdev_pos_ck(i_bus, i_contingency)) =e= 0;

Qg_hvdc_embedded_2_min_ck(i_hvdc_embedded_Q, i_contingency)..
Q_hvdc_embedded_2_ck(i_hvdc_embedded_Q, i_contingency) =g= hvdc_embedded_Qmin(i_hvdc_embedded_Q);

Qg_hvdc_embedded_2_max_ck(i_hvdc_embedded_Q, i_contingency)..
Q_hvdc_embedded_2_ck(i_hvdc_embedded_Q, i_contingency) =l= hvdc_embedded_Qmax(i_hvdc_embedded_Q);

hvdc_embedded_2_PQswitchMax(i_hvdc_embedded_Q, i_contingency)..
(Q_hvdc_embedded_2_ck(i_hvdc_embedded_Q, i_contingency) - hvdc_embedded_Qmax(i_hvdc_embedded_Q)) * sum(i_bus$(hvdc_embedded_map_Q2(i_hvdc_embedded_Q,i_bus)), Vdev_neg_ck(i_bus, i_contingency)) =e= 0;

hvdc_embedded_2_PQswitchMin(i_hvdc_embedded_Q, i_contingency)..
(Q_hvdc_embedded_2_ck(i_hvdc_embedded_Q, i_contingency) - hvdc_embedded_Qmin(i_hvdc_embedded_Q)) * sum(i_bus$(hvdc_embedded_map_Q2(i_hvdc_embedded_Q,i_bus)), Vdev_pos_ck(i_bus, i_contingency)) =e= 0;

Qg_hvdc_interconnection_min_ck(i_hvdc_interconnection, i_contingency)..
Q_hvdc_interconnection_ck(i_hvdc_interconnection, i_contingency) =g= hvdc_interconnection_Qmin(i_hvdc_interconnection);

Qg_hvdc_interconnection_max_ck(i_hvdc_interconnection, i_contingency)..
Q_hvdc_interconnection_ck(i_hvdc_interconnection, i_contingency) =l= hvdc_interconnection_Qmax(i_hvdc_interconnection);

hvdc_interconnection_PQswitchMax(i_hvdc_interconnection, i_contingency)..
(Q_hvdc_interconnection_ck(i_hvdc_interconnection, i_contingency) - hvdc_interconnection_Qmax(i_hvdc_interconnection)) * sum(i_bus$(hvdc_interconnection_map(i_hvdc_interconnection,i_bus)), Vdev_neg_ck(i_bus, i_contingency)) =e= 0;

hvdc_interconnection_PQswitchMin(i_hvdc_interconnection, i_contingency)..
(Q_hvdc_interconnection_ck(i_hvdc_interconnection, i_contingency) - hvdc_interconnection_Qmin(i_hvdc_interconnection)) * sum(i_bus$(hvdc_interconnection_map(i_hvdc_interconnection,i_bus)), Vdev_pos_ck(i_bus, i_contingency)) =e= 0;

Qg_hvdc_spit_min_ck(i_hvdc_spit, i_contingency)..
Q_hvdc_spit_ck(i_hvdc_spit, i_contingency) =g= hvdc_spit_Qmin(i_hvdc_spit);

Qg_hvdc_spit_max_ck(i_hvdc_spit, i_contingency)..
Q_hvdc_spit_ck(i_hvdc_spit, i_contingency) =l= hvdc_spit_Qmax(i_hvdc_spit);

hvdc_spit_PQswitchMax(i_hvdc_spit, i_contingency)..
(Q_hvdc_spit_ck(i_hvdc_spit, i_contingency) - hvdc_spit_Qmax(i_hvdc_spit)) * sum(i_bus$(hvdc_spit_map(i_hvdc_spit,i_bus)), Vdev_neg_ck(i_bus, i_contingency)) =e= 0;

hvdc_spit_PQswitchMin(i_hvdc_spit, i_contingency)..
(Q_hvdc_spit_ck(i_hvdc_spit, i_contingency) - hvdc_spit_Qmin(i_hvdc_spit)) * sum(i_bus$(hvdc_spit_map(i_hvdc_spit,i_bus)), Vdev_pos_ck(i_bus, i_contingency)) =e= 0;

Voltage_min_ck(i_bus, i_contingency)..
V_ck(i_bus, i_contingency) =g= 0.9;

Voltage_max_ck(i_bus, i_contingency)..
V_ck(i_bus, i_contingency) =l= 1.1;

Angles_min_ck(i_bus, i_contingency)..
theta_ck(i_bus, i_contingency) =g= -pi;

Angles_max_ck(i_bus, i_contingency)..
theta_ck(i_bus, i_contingency) =l= pi;

P_balance_ck(i_bus, i_contingency)..
sum(i_sync$(sync_map(i_sync, i_bus)), P_sync_ck(i_sync, i_contingency))
+ sum(i_wind$(wind_map(i_wind, i_bus)), P_wind(i_wind))
+ sum(i_hvdc_embedded, P_hvdc_embedded(i_hvdc_embedded)*hvdc_embedded_map(i_hvdc_embedded, i_bus))
+ sum(i_hvdc_interconnection$(hvdc_interconnection_map(i_hvdc_interconnection, i_bus)), P_hvdc_interconnection(i_hvdc_interconnection))
+ sum(i_hvdc_spit$(hvdc_spit_map(i_hvdc_spit, i_bus)), P_hvdc_spit(i_hvdc_spit))
- sum(i_dispatchable_load$(dispatchable_load_map(i_dispatchable_load, i_bus)), P_dispatchable_load(i_dispatchable_load))
- demand(i_bus)
=e=
sum(i_branch$(branch_map(i_branch,i_bus) = 1),P1_ck(i_branch, i_contingency))+sum(i_branch$(branch_map(i_branch,i_bus) = -1),P2_ck(i_branch, i_contingency));

Q_balance_ck(i_bus, i_contingency)..
sum(i_sync$(sync_map(i_sync, i_bus)), Q_sync_ck(i_sync, i_contingency))
+ sum(i_wind$(wind_map(i_wind, i_bus)), Q_wind_ck(i_wind, i_contingency))
+ sum(i_syncon$(syncon_map(i_syncon, i_bus)), Q_syncon_ck(i_syncon, i_contingency))
+ sum(i_statcom$(statcom_map(i_statcom, i_bus)), Q_statcom_ck(i_statcom, i_contingency))
+ sum(i_shunt$(shunt_map(i_shunt, i_bus)), Q_shunt(i_shunt)*V_ck(i_bus, i_contingency)*V_ck(i_bus, i_contingency))
- sum(i_statvar$(statvar_map(i_statvar, i_bus)), Q_statvar_ck(i_statvar, i_contingency))
+ sum(i_hvdc_embedded_Q, Q_hvdc_embedded_1_ck(i_hvdc_embedded_Q, i_contingency)*hvdc_embedded_map_Q1(i_hvdc_embedded_Q, i_bus))
+ sum(i_hvdc_embedded_Q, Q_hvdc_embedded_2_ck(i_hvdc_embedded_Q, i_contingency)*hvdc_embedded_map_Q2(i_hvdc_embedded_Q, i_bus))
+ sum(i_hvdc_interconnection$(hvdc_interconnection_map(i_hvdc_interconnection, i_bus)), Q_hvdc_interconnection_ck(i_hvdc_interconnection, i_contingency))
+ sum(i_hvdc_spit$(hvdc_spit_map(i_hvdc_spit, i_bus)), Q_hvdc_spit_ck(i_hvdc_spit, i_contingency))
- demandQ(i_bus)
=e=
sum(i_branch$(branch_map(i_branch,i_bus) = 1),Q1_ck(i_branch, i_contingency))+sum(i_branch$(branch_map(i_branch,i_bus) = -1),Q2_ck(i_branch, i_contingency));

line_P1_ck(i_branch, i_contingency)..
P1_ck(i_branch, i_contingency)
=e=
contingency_states(i_branch, i_contingency)*sum(i_bus$(branch_map(i_branch,i_bus) = 1),(sum(j_bus$(branch_map(i_branch,j_bus)=-1), V_ck(i_bus, i_contingency)*(V_ck(i_bus, i_contingency)*Gff(i_branch)+V_ck(j_bus, i_contingency)*Gft(i_branch)*cos(theta_ck(i_bus, i_contingency)-theta_ck(j_bus, i_contingency))+V_ck(j_bus, i_contingency)*Bft(i_branch)*sin(theta_ck(i_bus, i_contingency)-theta_ck(j_bus, i_contingency))))));

line_Q1_ck(i_branch, i_contingency)..
Q1_ck(i_branch, i_contingency)
=e=
contingency_states(i_branch, i_contingency)*sum(i_bus$(branch_map(i_branch,i_bus) = 1),(sum(j_bus$(branch_map(i_branch,j_bus)=-1), V_ck(i_bus, i_contingency)*(-V_ck(i_bus, i_contingency)*Bff(i_branch)+V_ck(j_bus, i_contingency)*Gft(i_branch)*sin(theta_ck(i_bus, i_contingency)-theta_ck(j_bus, i_contingency))-V_ck(j_bus, i_contingency)*Bft(i_branch)*cos(theta_ck(i_bus, i_contingency)-theta_ck(j_bus, i_contingency))))));

line_P2_ck(i_branch, i_contingency)..
P2_ck(i_branch, i_contingency)
=e=
contingency_states(i_branch, i_contingency)*sum(i_bus$(branch_map(i_branch,i_bus) = -1),(sum(j_bus$(branch_map(i_branch,j_bus)=1), V_ck(i_bus, i_contingency)*(V_ck(i_bus, i_contingency)*Gtt(i_branch)+V_ck(j_bus, i_contingency)*Gtf(i_branch)*cos(theta_ck(i_bus, i_contingency)-theta_ck(j_bus, i_contingency))+V_ck(j_bus, i_contingency)*Btf(i_branch)*sin(theta_ck(i_bus, i_contingency)-theta_ck(j_bus, i_contingency))))));

line_Q2_ck(i_branch, i_contingency)..
Q2_ck(i_branch, i_contingency)
=e=
contingency_states(i_branch, i_contingency)*sum(i_bus$(branch_map(i_branch,i_bus) = -1),(sum(j_bus$(branch_map(i_branch,j_bus)=1), V_ck(i_bus, i_contingency)*(-V_ck(i_bus, i_contingency)*Btt(i_branch)+V_ck(j_bus, i_contingency)*Gtf(i_branch)*sin(theta_ck(i_bus, i_contingency)-theta_ck(j_bus, i_contingency))-V_ck(j_bus, i_contingency)*Btf(i_branch)*cos(theta_ck(i_bus, i_contingency)-theta_ck(j_bus, i_contingency))))));

line_max1_ck(i_branch, i_contingency)..
P1_ck(i_branch, i_contingency)*P1_ck(i_branch, i_contingency)+Q1_ck(i_branch, i_contingency)*Q1_ck(i_branch, i_contingency) =l= branch_max_E(i_branch)*branch_max_E(i_branch);

line_max2_ck(i_branch, i_contingency)..
P2_ck(i_branch, i_contingency)*P2_ck(i_branch, i_contingency)+Q2_ck(i_branch, i_contingency)*Q2_ck(i_branch, i_contingency) =l= branch_max_E(i_branch)*branch_max_E(i_branch);


***************************************************************
*** SOLVE
***************************************************************

model test /all/;

*option reslim = 600;
option nlp=ipopt;
test.optfile=1;

solve test using nlp minimizing deviation;

scalar sol;
sol = test.modelstat;

execute_unload 'PostPSCACOPF' deviation, P_sync, P_wind, P_hvdc_embedded, P_hvdc_interconnection, P_hvdc_spit, P_dispatchable_load, Q_sync, Q_wind, Q_syncon, Q_statcom, Q_shunt, Q_statvar, Q_hvdc_embedded_1, Q_hvdc_embedded_2, Q_hvdc_interconnection, Q_hvdc_spit, V, theta, sol, V_ck, DeltaF_ck, B4_flow, B6_flow;
