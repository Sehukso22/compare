
from lshashing import LSHRandom
import numpy as np
import math
import csv
import sys, getopt
import os.path
from os import path
from time import perf_counter_ns, sleep
import random
import yaml
from pykdtree.kdtree import KDTree
from sklearn.neighbors import BallTree
import hnswlib
from pynndescent import NNDescent
from annoy import AnnoyIndex
import os, psutil
import winsound
from covid_data.covid_parser import get_data


LSH = 'lsh'
PYKDTREE = 'pykdtree'
HNSW = 'hnsw'
PYNNDESCENT = 'pynndescent'
BALL_TREE = 'balltree'
ANNOY = 'annoy'
methods = [LSH, PYKDTREE, HNSW, PYNNDESCENT, BALL_TREE, ANNOY]
recalls = {}
queries = {}
options = {'input_file' : '', 'data_source' : 'generator', 'query' : 1, 'naive' : False, 'construction': 1, 'k' : [1], 'methods' : [], 'verbose' : False}

def parse_arguments(argv, options):
    opts, args = getopt.getopt(argv,'i:d:q:nc:k:m:vh',['input=','data=','query=', 'naive', 'construction=', 'neighbours=', 'methods=', 'verbose', 'help'])
    for opt, arg in opts:
        if opt in ('-i', '--input'):
            options['data_file'] = f'generated_data/points_{arg}.csv'
            if not path.exists(options['data_file']):
                sys.exit(f'The file {options["data_file"]} does not exist. Generate the file with generator.py')
        elif opt in ('-d', '--data'):
            if arg not in ('covid', 'generator'):
                sys.exit('Option data must be either data or generator')
            options['data_source'] = arg
        elif opt in ('-q', '--query'):
            options['query'] = int(arg)
            if not options['query'] > 0:
                sys.exit('Invalid number of queries')
        elif opt in ('-n', '--naive'):
            options['naive'] = True
        elif opt in ('-c', '--construction'):
            options['construction'] = int(arg)
            if not options['construction'] > 0:
                sys.exit('Invalid number of constructions')
        elif opt in ('-k', '--neighbours'):
            options['k'] = arg.split(',')
            for k in options['k']:
                if not int(k) > 0:
                    sys.exit('Invalid number of neighbors to search for')
        elif opt in ('-m', '--methods'):
            options['methods'] = arg.split(',')
            for method in options['methods']:
                if method not in methods:
                    sys.exit(f'Invalid method. List of allowed methods: {", ".join(methods)}')
        elif opt in ('-v', '--verbose'):
            options['verbose'] = True
        elif opt in ('-h', '--help'):
            print('Possible options are:')
            print('-d, --data: Possible values are "covid" or "generator". Default is "generator".');
            print('-i, --input: Generated input file. Only the name of the file without suffix or path is needed. For instance "10000_10" will reffer to the file "generated_data/10000_10.csv"');
            print('-c, --construction: Number of constructions. Default is 1.');
            print('-q, --query: Number of queries. Default is 1.');
            print('-k Parameter k of KNN. multiple values can be provided, divided by comma like "4,10,25". Default is 1.');
            print('-m, --methods: Methods to be used. Names divided by comma like "hnsw,balltree". Supported methods are lsh, pykdtree, hnsw, pynndescent, balltree, annoy');
            print('-v, --verbose: Verbose');
            print('-n, --naive: Print naive results');
            sys.exit()

def get_generator_data():
    data = []
    with open(options['data_file'], 'r') as file:
        reader = csv.reader(file,delimiter=',')
        for row in reader:
            data.append([eval(i) for i in row])
    return np.array(data)

def euclidean(v1, v2):
    d = ((v1 - v2) ** 2).sum()
    return math.sqrt(d)

def get_knn_naive(points, query_point, k):
    result = []
    for i in range(points.shape[0]):
        ip = points[i, :]
        x = (euclidean(ip, query_point), i, ip)
        result.append(x)
    return sorted(result, key = lambda x: x[0])[:k]

def log(label, data=None):
    if options['verbose']:
        if not data is None:
            print(label, data)
        else:
            print(label)

def extract_query_points(data_points, number_of_points):
    if number_of_points >= len(data_points):
        sys.exit('Number of neighbors to search is too high for the data set')
    step = int(len(data_points) / number_of_points)
    query_points = data_points[0::step]
    del data_points[0::step]
    return np.array(data_points), np.array(query_points)

parse_arguments(sys.argv[1:], options)

for k_string in options['k']:

    recalls[k_string] = {
        'lsh_recall': 0,
        'kdtree_recall': 0,
        'pykdtree_recall': 0,
        'ball_tree_recall': 0,
        'hnsw_recall': 0,
        'pynndescent_recall': 0,
        'annoy_recall': 0
    }
    
    queries[k_string] = {
        'lsh_query_time': 0,
        'kdtree_query_time': 0,
        'pykdtree_query_time': 0,
        'ball_tree_query_time': 0,
        'hnsw_query_time': 0,
        'pynndescent_query_time': 0,
        'annoy_query_time': 0,
        'naive_query_time': 0,
    }


if options['data_source'] == 'generator':
    data_points = get_generator_data()
else:
    data_points = get_data()

data_points, query_points = extract_query_points(data_points.tolist(), options['query'])
log('construction')

if LSH in options['methods']:
    lsh_construction_time = 0
    lsh_memory = 0
    lsh_memory2 = 0
    with open('methods_config/lsh.yaml', 'r') as stream:
        lsh_configuration = yaml.safe_load(stream)
    lsh_num_tables = lsh_configuration['tables']
    lsh_hash_len = lsh_configuration['hash_length']
    lsh_radius = lsh_configuration['radius']
    lsh_buckets = lsh_configuration['buckets']
    
    for i in range(0, options['construction']):
        start_memory = psutil.Process(os.getpid()).memory_info().rss
        start = perf_counter_ns()
        lshashing = LSHRandom(data_points, hash_len = lsh_hash_len, num_tables = lsh_num_tables)
        end = perf_counter_ns()
        memory = psutil.Process(os.getpid()).memory_info().rss - start_memory
        memory2 = total_size(lshashing)
        time = end - start
        log('lsh construction time:', time)
        log('lsh size:', memory)
        lsh_construction_time = lsh_construction_time + time
        lsh_memory = lsh_memory + memory
        lsh_memory2 = lsh_memory2 + memory2
        
    # check for bug in lshashing library, try 100 cosntructions. If the lsh tables have less hash keys than query parameter buckets, the search stucks in infinite loop
    lsh_construction_try = 0
    error = True
    while (error):
        if lsh_construction_try > 100:
            sys.exit('Unable to create LSH tables with number of keys to satisfy "buckets" searching parameter')
        lsh_construction_try = lsh_construction_try + 1
        error = False
        for table in lshashing.tables:
            if len(table.hash_table.keys()) < lsh_configuration['buckets']:
                log('LSH error, number of hashes in table:', len(table.hash_table.keys()))
                error = True
                break
        if (error):
            lshashing = LSHRandom(data_points, hash_len = lsh_hash_len, num_tables = lsh_num_tables)
if PYKDTREE in options['methods']:
    pykdtree_construction_time = 0
    pykdtree_memory = 0
    with open('methods_config/pykdtree.yaml', 'r') as stream:
        pykdtree_configuration = yaml.safe_load(stream)
        pykdtree_epsilon = pykdtree_configuration['eps']
        pykdtree_leaf_size = pykdtree_configuration['leafsize']
    for i in range(0, options['construction']):
        start_memory = psutil.Process(os.getpid()).memory_info().rss
        start = perf_counter_ns()
        pykdtree_obj = KDTree(data_points, leafsize=pykdtree_leaf_size)
        end = perf_counter_ns()
        memory = psutil.Process(os.getpid()).memory_info().rss - start_memory
        time = end - start
        log('pykdtree construction time:', time)
        log('pykdtree size:', memory)
        pykdtree_construction_time = pykdtree_construction_time + time
        pykdtree_memory = pykdtree_memory + memory
if BALL_TREE in options['methods']:
    ball_tree_construction_time = 0
    ball_tree_memory = 0
    with open('methods_config/balltree.yaml', 'r') as stream:
        ball_tree_configuration = yaml.safe_load(stream)
        ball_tree_leaf_size = ball_tree_configuration['leaf_size']
        ball_tree_breadth_first = ball_tree_configuration['breadth_first']
    for i in range(0, options['construction']):
        start_memory = psutil.Process(os.getpid()).memory_info().rss
        start = perf_counter_ns()
        ball_tree_obj = BallTree(data_points, leaf_size=ball_tree_leaf_size)  
        end = perf_counter_ns()
        memory = psutil.Process(os.getpid()).memory_info().rss - start_memory
        time = end - start
        log('ball tree construction time:', time)
        log('ball tree size:', memory)
        ball_tree_construction_time = ball_tree_construction_time + time
        ball_tree_memory = ball_tree_memory + memory
if HNSW in options['methods']:
    hnsw_construction_time = 0
    hnsw_memory = 0
    with open('methods_config/hnsw.yaml', 'r') as stream:
        hnsw_configuration = yaml.safe_load(stream)
        hnsw_ef_construction =hnsw_configuration['ef_construction']
        hnsw_M = hnsw_configuration['M']
        hnsw_ef =hnsw_configuration['ef']
    for i in range(0, options['construction']):
        start_memory = psutil.Process(os.getpid()).memory_info().rss
        start = perf_counter_ns()
        hnsw_obj = hnswlib.Index('l2', dim = data_points.shape[1])  
        hnsw_obj.init_index(max_elements = len(data_points), ef_construction = hnsw_ef_construction, M = hnsw_M)
        hnsw_obj.add_items(data_points)
        end = perf_counter_ns()
        memory = psutil.Process(os.getpid()).memory_info().rss - start_memory
        time = end - start
        log('hnsw graph construction time:', time)
        log('hnsw graph size:', memory)
        hnsw_construction_time = hnsw_construction_time + time
        hnsw_memory = hnsw_memory + memory
if PYNNDESCENT in options['methods']:
    pynndescent_construction_time = 0
    pynndescent_memory = 0
    with open('methods_config/pynndescent.yaml', 'r') as stream:
        pynndescent_configuration = yaml.safe_load(stream)
        pynndescent_pruning_degree_multiplier = pynndescent_configuration['pruning_degree_multiplier']
        pynndescent_low_memory = pynndescent_configuration['low_memory']
        pynndescent_n_iters = pynndescent_configuration['n_iters']
        pynndescent_delta = pynndescent_configuration['delta']
        pynndescent_n_neighbors = pynndescent_configuration['n_neighbors']
    for i in range(0, options['construction']):
        start_memory = psutil.Process(os.getpid()).memory_info().rss
        start = perf_counter_ns()
        pynndescent_obj = NNDescent(data_points, tree_init=False, pruning_degree_multiplier=pynndescent_pruning_degree_multiplier, low_memory=pynndescent_low_memory, n_iters= pynndescent_n_iters, delta=pynndescent_delta,n_neighbors=pynndescent_n_neighbors)
        end = perf_counter_ns()
        memory = psutil.Process(os.getpid()).memory_info().rss - start_memory
        time = end - start
        log('pynndescent graph construction time:', time)
        log('pynndescent graph size:', memory)
        pynndescent_construction_time = pynndescent_construction_time + time
        pynndescent_memory = pynndescent_memory + memory
if ANNOY in options['methods']:
    annoy_construction_time = 0
    annoy_memory = 0
    with open('methods_config/annoy.yaml', 'r') as stream:
        annoy_configuration = yaml.safe_load(stream)
        annoy_n_trees = annoy_configuration['n_trees']
        annoy_search_k = annoy_configuration['search_k']
    for i in range(0, options['construction']):
        start_memory = psutil.Process(os.getpid()).memory_info().rss
        start = perf_counter_ns()
        annoy_obj = AnnoyIndex(f = data_points.shape[1], metric='euclidean')  
        for index, data_point in enumerate(data_points):
            annoy_obj.add_item(index, data_point);
        annoy_obj.build(annoy_n_trees, n_jobs=1)
        end = perf_counter_ns()
        memory = psutil.Process(os.getpid()).memory_info().rss - start_memory
        time = end - start
        log('annoy construction time:', time)
        log('annoy size:', memory)
        annoy_construction_time = annoy_construction_time + time
        annoy_memory = annoy_memory + memory
log('')
log('query')


# first search in pynndescent takes very long time. Skip it
first_pynndescent = True

for k_string in options['k']:
    k = int(k_string)
    for point in query_points:
        start = perf_counter_ns()
        naive = get_knn_naive(data_points, point, k)
        end = perf_counter_ns()
        if options['naive']:
            time = end - start
            log('naive time:', time)
            queries[k_string]['naive_query_time'] = queries[k_string]['naive_query_time'] + time

        if LSH in options['methods']:
            start = perf_counter_ns()
            lsh_result = lshashing.knn_search(data_points, point, k = k, buckets = lsh_buckets, radius = lsh_radius)
            end = perf_counter_ns()
            time = end - start
            log('lsh time:', time)
            queries[k_string]['lsh_query_time'] = queries[k_string]['lsh_query_time'] + time
            found = 0
            for l in lsh_result:
                for _, index, _ in naive:
                    if l[1] == index:
                        found = found + 1
            recall = found/k
            log('lsh recall:', recall)
            recalls[k_string]['lsh_recall'] = recalls[k_string]['lsh_recall'] + recall
        
        if PYKDTREE in options['methods']:
            search = np.array([point])
            start = perf_counter_ns()
            pykdtree_results = pykdtree_obj.query(search, k=k, eps=pykdtree_epsilon) 
            end = perf_counter_ns()
            time = end - start
            log('pykdtree time:', time)
            queries[k_string]['pykdtree_query_time'] = queries[k_string]['pykdtree_query_time'] + time
            found = 0
            if k == 1:
                to_iterate = pykdtree_results[1]
            else:
                to_iterate = pykdtree_results[1][0]
                
            for pykdtree_result_index in to_iterate:
                for _, index, _ in naive:
                    if pykdtree_result_index == index:
                        found = found + 1
            recall = found/k
            log('pykdtree recall:', recall)
            recalls[k_string]['pykdtree_recall'] = recalls[k_string]['pykdtree_recall'] + recall

        if BALL_TREE in options['methods']:
            search = np.array([point])
            start = perf_counter_ns()
            ball_tree_results = ball_tree_obj.query(search, k=k, breadth_first=ball_tree_breadth_first) 
            end = perf_counter_ns()
            time = end - start
            log('ball tree time:', time)
            queries[k_string]['ball_tree_query_time'] = queries[k_string]['ball_tree_query_time'] + time
            found = 0
            for ball_tree_result_index in ball_tree_results[1][0]:
                for _, index, _ in naive:
                    if ball_tree_result_index == index:
                        found = found + 1
            recall = found/k
            log('ball tree recall:', recall)
            recalls[k_string]['ball_tree_recall'] = recalls[k_string]['ball_tree_recall'] + recall
               
        if HNSW in options['methods']:
            search = np.array([point])
            if hnsw_ef > k:
                hnsw_obj.set_ef(hnsw_ef)
            else:
                hnsw_obj.set_ef(k * 2)            
            start = perf_counter_ns()
            hnsw_results = hnsw_obj.knn_query(search, k=k) 
            end = perf_counter_ns()
            time = end - start
            log('hnsw graph time:', time)
            queries[k_string]['hnsw_query_time'] = queries[k_string]['hnsw_query_time'] + time
            found = 0
            for hnsw_result_index in hnsw_results[0][0]:
                for _, index, _ in naive:
                    if hnsw_result_index == index:
                        found = found + 1
            recall = found/k
            log('hnsw graph recall:', recall)
            recalls[k_string]['hnsw_recall'] = recalls[k_string]['hnsw_recall'] + recall

        if PYNNDESCENT in options['methods']:
            search = np.array([point])
            if first_pynndescent:
                first_pynndescent = False
                pynndescent_obj.query(search, k=k) 
            start = perf_counter_ns()
            pynndescent_results = pynndescent_obj.query(search, k=k) 
            end = perf_counter_ns()
            time = end - start
            log('pynndescent graph time:', time)
            queries[k_string]['pynndescent_query_time'] = queries[k_string]['pynndescent_query_time'] + time
            found = 0
            for pynndescent_result_index in pynndescent_results[0][0]:
                for _, index, _ in naive:
                    if pynndescent_result_index == index:
                        found = found + 1
            recall = found/k
            log('pynndescent graph recall:', recall)
            recalls[k_string]['pynndescent_recall'] = recalls[k_string]['pynndescent_recall'] + recall

        if ANNOY in options['methods']:
            search = np.array(point)
            start = perf_counter_ns()
            annoy_results = annoy_obj.get_nns_by_vector(search, n=k, search_k=annoy_search_k) 
            end = perf_counter_ns()
            time = end - start
            log('annoy graph time:', time)
            queries[k_string]['annoy_query_time'] = queries[k_string]['annoy_query_time'] + time
            found = 0
            for annoy_result_index in annoy_results:
                for _, index, _ in naive:
                    if annoy_result_index == index:
                        found = found + 1
            recall = found/k
            log('annoy graph recall:', recall)
            recalls[k_string]['annoy_recall'] = recalls[k_string]['annoy_recall'] + recall
           
            
log('')
print('summary')
print('dataset source:', options['data_source'])
print('points:', len(data_points))
print('dimensions:', data_points.shape[1])
print('')

if LSH in options['methods']:
    print('')
    print('lsh tables:', lsh_num_tables)
    print('lsh hash length:', lsh_hash_len)
    print('lsh construction time:', lsh_construction_time / options['construction'])
    print('lsh memory usage:', lsh_memory2 / options['construction'])

if PYKDTREE in options['methods']:
    print('')
    print('pykdtree epsilon:', pykdtree_epsilon)
    print('pykdtree leaf size:', pykdtree_leaf_size)
    print('pykdtree construction time:', pykdtree_construction_time / options['construction'])
    print('pykdtree memory usage:', pykdtree_memory / options['construction'])
    
if BALL_TREE in options['methods']:
    print('')
    print('ball tree leaf size:', ball_tree_leaf_size)
    print('ball tree breadth first:', ball_tree_breadth_first)
    print('ball tree construction time:', ball_tree_construction_time / options['construction'])
    print('ball tree memory usage:', ball_tree_memory / options['construction'])

if HNSW in options['methods']:
    print('')
    print('hnsw M:', hnsw_M)
    print('hnsw ef construction:', hnsw_ef_construction)
    print('hnsw ef:', hnsw_ef)
    print('hnsw graph construction time:', hnsw_construction_time / options['construction'])
    print('hnsw graph memory usage:', hnsw_memory / options['construction'])

if PYNNDESCENT in options['methods']:
    print('')
    print('pynndescent low memory:', pynndescent_low_memory)
    print('pynndescent delta:', pynndescent_delta)
    print('pynndescent n iters:', pynndescent_n_iters)
    print('pynndescent n neighbors:', pynndescent_n_neighbors)
    print('pynndescent graph construction time:', pynndescent_construction_time / options['construction'])
    print('pynndescent graph memory usage:', pynndescent_memory / options['construction'])
if ANNOY in options['methods']:
    print('')
    print('annoy n trees:', annoy_n_trees)
    print('annoy search k:', annoy_search_k)
    print('annoy graph construction time:', annoy_construction_time / options['construction'])
    print('annoy graph memory usage:', annoy_memory / options['construction'])


for k_string in options['k']:
    print('')
    print('k:', k_string)
    if options['naive']:
        print('naive query time:', queries[k_string]['naive_query_time'] / len(query_points))

    if LSH in options['methods']:
        print('')
        print('lsh query time:', queries[k_string]['lsh_query_time'] / len(query_points))
        print('lsh recall:', recalls[k_string]['lsh_recall'] / len(query_points))
             
    if PYKDTREE in options['methods']:
        print('')
        print('pykdtree query time:', queries[k_string]['pykdtree_query_time'] / len(query_points))
        print('pykdtree recall:', recalls[k_string]['pykdtree_recall'] / len(query_points))
        
    if BALL_TREE in options['methods']:
        print('')
        print('ball tree query time:', queries[k_string]['ball_tree_query_time'] / len(query_points))
        print('ball tree recall:', recalls[k_string]['ball_tree_recall'] / len(query_points))

    if HNSW in options['methods']:
        print('')
        print('hnsw graph query time:', queries[k_string]['hnsw_query_time'] / len(query_points))
        print('hnsw graph recall:', recalls[k_string]['hnsw_recall'] / len(query_points))

    if PYNNDESCENT in options['methods']:
        print('')
        print('pynndescent graph query time:', queries[k_string]['pynndescent_query_time'] / len(query_points))
        print('pynndescent graph recall:', recalls[k_string]['pynndescent_recall'] / len(query_points))
     
    if ANNOY in options['methods']:
        print('')
        print('annoy graph query time:', queries[k_string]['annoy_query_time'] / len(query_points))
        print('annoy graph recall:', recalls[k_string]['annoy_recall'] / len(query_points))

winsound.MessageBeep()