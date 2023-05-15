Following dependencies needs to be installed as modules:
https://github.com/storpipfugl/pykdtree
https://github.com/scikit-learn/scikit-learn
https://github.com/MNoorFawi/lshashing
https://github.com/spotify/annoy
https://github.com/nmslib/hnswlib
https://github.com/lmcinnes/pynndescent

You can use script install.sh that uses pip to install these dependencies.

Usage:
python generator.py - generates artificial data for experiments. The script uses generator_config.yaml

Python compare.py - runs experiments

use the compare.py with following options:

-d, --data: Possible values are "covid" or "generator". Default is "generator".
-i, --input: Generated input file. Only the name of the file without suffix or path is needed. For instance "10000_10" will reffer to the file "generated_data/10000_10.csv"
-c, --construction: Number of constructions. Default is 1.
-q, --query: Number of queries. Default is 1.
-k Parameter k of KNN. multiple values can be provided, divided by comma like "4,10,25". Default is 1.
-m, --methods: Methods to be used. Names divided by comma like "hnsw,balltree". Supported methods are lsh, pykdtree, hnsw, pynndescent, balltree, annoy
-v, --verbose: Verbose
-n, --naive: Print naive results