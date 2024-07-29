#!/bin/bash
grep -irl "/home/fkli" ./ | xargs sed -i "s/\/home\/fkli/\/lustre\/home\/fkli/g"