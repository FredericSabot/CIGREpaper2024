cd dynawo_files

sed -i 's/lib="dynawo_SolverSIM"/lib="dynawo_SolverIDA"/g' *
sed -i 's/parId="SolverSIM"/parId="IDAOrder2"/g' *