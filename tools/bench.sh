#!/usr/bin/env bash
# 运行前： chmod +x tools/bench.sh

echo "proc,wall" > bench.csv
for p in 1 2 4 8; do
  t=$( { /usr/bin/time -p mpirun -n $p ./bs_price \
         ../examples/params.json --scheme cn >/dev/null; } 2>&1 | awk '/real/{print $2}' )
  echo "$p,$t" | tee -a bench.csv
done
mv bench.csv ../tools/

