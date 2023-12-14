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
set i_hvdc_spit hvdc spit;
set i_dispatchable_load hvdc dispatchable loads;

***************************************************************
*** PARAMETERS
***************************************************************

parameter Epsilon;
Epsilon = 1e-4;

parameter Kg;
Kg = 1000;

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

parameter P_sync_0(i_sync) initial sync outputs;
parameter P_wind_0(i_wind) initial wind outputs;
parameter P_hvdc_embedded_0(i_hvdc_embedded) initial embedded hvdc flows;
parameter P_hvdc_interconnection_0(i_hvdc_interconnection) initial hvdc interconection flows;
parameter P_hvdc_spit_0(i_hvdc_spit) initial hvdc interconection flows;
parameter P_dispatchable_load_0(i_dispatchable_load) initial dispatchable load demand;
parameter Ppf_0(i_branch) initial line active power flows;

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
parameter hvdc_spit_Qmin(i_hvdc_spit) spit hvdc minimum reactive generation;
parameter hvdc_spit_Qmax(i_hvdc_spit) spit hvdc maximum reactive generation;
parameter dispatchable_load_min(i_dispatchable_load) dispatchable load minimum demand;
parameter dispatchable_load_max(i_dispatchable_load) dispatchable load maximum demand;

*LINES DATA

parameter branch_map(i_branch,i_bus) line map;
parameter G(i_bus,i_bus) conductance matrix;
parameter B(i_bus,i_bus) susceptance matrix;
parameter Gff(i_branch) line conductance (from-from);
parameter Gft(i_branch) line conductance (from-to);
parameter Bff(i_branch) line susceptance (from-from);
parameter Bft(i_branch) line susceptance (from-to);

parameter branch_max_N(i_branch) line capacities;

*DEMAND DATA

parameter demand(i_bus) active load at bus s;
parameter demandQ(i_bus) reactive load at bus s;


$gdxin PreACOPF
$load i_sync i_wind i_syncon i_statcom i_shunt i_statvar i_bus i_branch i_hvdc_embedded i_hvdc_embedded_Q i_hvdc_interconnection i_hvdc_spit i_dispatchable_load sync_map wind_map syncon_map statcom_map shunt_map statvar_map hvdc_embedded_map hvdc_embedded_map_Q1 hvdc_embedded_map_Q2 hvdc_interconnection_map hvdc_spit_map dispatchable_load_map lincost hvdc_interconnection_costs dispatchable_load_costs P_sync_0 P_wind_0 P_hvdc_embedded_0 P_hvdc_interconnection_0 P_hvdc_spit_0 P_dispatchable_load_0 sync_min sync_max sync_Qmin sync_Qmax wind_max wind_Qmin wind_Qmax syncon_Qmin syncon_Qmax statcom_Qmin statcom_Qmax shunt_Qmin shunt_Qmax statvar_Qmin statvar_Qmax hvdc_embedded_min hvdc_embedded_max hvdc_embedded_Qmin hvdc_embedded_Qmax hvdc_interconnection_min hvdc_interconnection_max hvdc_interconnection_Qmin hvdc_interconnection_Qmax hvdc_spit_min hvdc_spit_max hvdc_spit_total_max hvdc_spit_Qmin hvdc_spit_Qmax dispatchable_load_min dispatchable_load_max demand demandQ G B Gff Gft Bff Bft branch_map branch_max_N Ppf_0
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
variable P_hvdc_spit(i_hvdc_spit) power setpoint of hvdc spit
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
variable Q_hvdc_spit(i_hvdc_spit) reactive power setpoint of hvdc spit

variable V(i_bus) bus voltage amplitude
variable theta(i_bus) bus voltage angles

variable Ppf(i_branch) active power flow through lines
variable Qpf(i_branch) reactive power flow through lines

variable pf(i_branch) power flow through lines

* positive variable load_shedding(i_bus) load shedding for relaxation and identification of problematic buses

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
Pg_hvdc_spit_min(i_hvdc_spit) minimum hvdc interconection active output
Pg_hvdc_spit_max(i_hvdc_spit) maximum hvdc interconection active output
Pg_hvdc_spit_total_max maximum wind available for spit hvdcs
Pg_dispatchable_load_min(i_dispatchable_load) minimum dispatchable load active dcemand
Pg_dispatchable_load_max(i_dispatchable_load) maximum dispatchable load active dcemand
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
Voltage_angles_min(i_bus) voltage angles negative limit
Voltage_angles_max(i_bus) voltage angles positive limit
line_Pflow(i_branch) defining power flow through lines
line_Qflow(i_branch) defining power flow through lines
line_flow(i_branch) defining power flow through lines
line_capacity(i_branch) line capacitiy limit;


***************************************************************
*** SETTINGS
***************************************************************

*setting the reference bus
theta.fx ('27') = 0;
theta.l(i_bus)=0;

V.l(i_bus)=1;

P_sync.l(i_sync) = P_sync_0(i_sync);
P_wind.l(i_wind) = P_wind_0(i_wind);
P_hvdc_embedded.l(i_hvdc_embedded) = P_hvdc_embedded_0(i_hvdc_embedded);
P_hvdc_interconnection.l(i_hvdc_interconnection) = P_hvdc_interconnection_0(i_hvdc_interconnection);
P_hvdc_spit.l(i_hvdc_spit) = P_hvdc_spit_0(i_hvdc_spit);
P_dispatchable_load.l(i_dispatchable_load) = P_dispatchable_load_0(i_dispatchable_load);

Ppf.l(i_branch) = Ppf_0(i_branch);

*needed for running twice through the same set in a single equation
alias(i_bus, jb);


***************************************************************
*** EQUATIONS
***************************************************************

dev..
deviation =e=
sum(i_sync, power(P_sync(i_sync) - P_sync_0(i_sync), 2))
+ sum(i_wind, power(P_wind(i_wind) - P_wind_0(i_wind), 2))
+ sum(i_hvdc_embedded, power(P_hvdc_embedded(i_hvdc_embedded) - P_hvdc_embedded_0(i_hvdc_embedded), 2))
+ sum(i_hvdc_interconnection, power(P_hvdc_interconnection(i_hvdc_interconnection) - P_hvdc_interconnection_0(i_hvdc_interconnection), 2))
+ sum(i_dispatchable_load, power(P_dispatchable_load(i_dispatchable_load) - P_dispatchable_load_0(i_dispatchable_load), 2))
+ sum(i_hvdc_spit, power(P_hvdc_spit(i_hvdc_spit) - P_hvdc_spit_0(i_hvdc_spit), 2))
+ Kg * Q_penalty;
* + 1e4 * sum(i_bus, load_shedding(i_bus));

cost_eq..
cost =e= sum(i_sync, P_sync(i_sync) * lincost(i_sync))
       + sum(i_hvdc_interconnection, P_hvdc_interconnection(i_hvdc_interconnection) * hvdc_interconnection_costs(i_hvdc_interconnection))
       - sum(i_dispatchable_load, P_dispatchable_load(i_dispatchable_load) * dispatchable_load_costs(i_dispatchable_load));

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
- demand(i_bus)
* + load_shedding(i_bus)
- sum(i_dispatchable_load$(dispatchable_load_map(i_dispatchable_load, i_bus)), P_dispatchable_load(i_dispatchable_load))
=e=
V(i_bus) * sum(jb,V(jb) * (G(i_bus,jb) * cos(theta(i_bus)-theta(jb)) + B(i_bus,jb) * sin(theta(i_bus)-theta(jb))));

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
* + load_shedding(i_bus)
=e=
V(i_bus) * sum(jb,V(jb) * (G(i_bus,jb) * sin(theta(i_bus)-theta(jb)) - B(i_bus,jb) * cos(theta(i_bus)-theta(jb))));

Voltage_min(i_bus)..
V(i_bus) =g= 0.98;

Voltage_max(i_bus)..
V(i_bus) =l= 1.05;

Voltage_angles_min(i_bus)..
theta(i_bus) =g= -pi;

Voltage_angles_max(i_bus)..
theta(i_bus) =l= pi;

line_Pflow(i_branch)..
Ppf(i_branch) =e=
sum(i_bus$(branch_map(i_branch,i_bus) = 1), (sum(jb$(branch_map(i_branch,jb)=-1), V(i_bus) * V(jb) * (Gft(i_branch) * cos(theta(i_bus)-theta(jb)) + Bft(i_branch) * sin(theta(i_bus)-theta(jb))) + Gff(i_branch) * power(V(i_bus), 2) )));

line_Qflow(i_branch)..
Qpf(i_branch) =e=
sum(i_bus$(branch_map(i_branch,i_bus) = 1),(sum(jb$(branch_map(i_branch,jb)=-1), V(i_bus) * V(jb) * (Gft(i_branch) * sin(theta(i_bus)-theta(jb)) - Bft(i_branch) * cos(theta(i_bus)-theta(jb))) - Bff(i_branch) * power(V(i_bus), 2) )));

line_flow(i_branch)..
pf(i_branch) =e= Ppf(i_branch) * Ppf(i_branch) + Qpf(i_branch) * Qpf(i_branch);

line_capacity(i_branch)..
pf(i_branch) =l= branch_max_N(i_branch) * branch_max_N(i_branch);

***************************************************************
*** SOLVE
***************************************************************

model test /all/;

option nlp=ipopt;
test.optfile=1;
solve test using nlp minimizing deviation;

scalar sol;
sol = test.modelstat;

execute_unload 'PostACOPF' deviation, cost, Q_penalty, P_sync, Q_sync, P_wind, Q_wind, Q_syncon, Q_statcom, Q_shunt, Q_statvar, P_hvdc_embedded, Q_hvdc_embedded_1, Q_hvdc_embedded_2, P_hvdc_interconnection, Q_hvdc_interconnection, P_hvdc_spit, Q_hvdc_spit, P_dispatchable_load, V, theta, pf, sol;
