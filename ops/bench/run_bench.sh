#!/bin/bash

echo "Running performance benchmarks..."

# Basic performance test
start_time=$(date +%s%N)

# Simulate some work
for i in {1..1000}; do
  echo -n > /dev/null
done

end_time=$(date +%s%N)
duration=$(( (end_time - start_time) / 1000000 ))

echo "Benchmark Results:"
echo "==================="
echo "Basic loop test: ${duration}ms"
echo "Memory usage: $(free -m | awk 'NR==2{printf "%.1f%%", $3*100/$2 }')"
echo "CPU usage: $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')"

# Test file I/O performance
echo "File I/O test..."
dd if=/dev/zero of=/tmp/testfile bs=1M count=10 2>&1 | grep copied

rm -f /tmp/testfile

echo "Benchmarks completed successfully"
