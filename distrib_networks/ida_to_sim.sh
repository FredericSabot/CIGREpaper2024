cd dynawo_files

sed -i 's/lib="dynawo_SolverIDA"/lib="dynawo_SolverSIM"/g' *
sed -i 's/parId="IDAOrder2"/parId="SolverSIM"/g' *