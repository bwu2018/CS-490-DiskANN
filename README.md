# CS-490-DiskANN

## Setup

#### Build EFANNA
*From proj root

```
cd efanna_imp/
cmake .
make
```

#### Build NSG

```
sudo apt-get install g++ cmake libboost-dev libgoogle-perftools-dev
```

*From proj root

```
cd nsg-imp/
mkdir build/ && cd build/
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

#### Download Datasets

*From proj root
```
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -zxvf sift.tar.gz
```

#### Setup shard dirs
*From proj root
```
mkdir shards/ && cd shards/
mkdir graphs
mkdir nsg_indexes
```

## Usage

```
cd disk-ann
```

### Make K-shards

```
python3 make_shards.py N_SHARDS N_CLOSEST_CENTERS N_DIM CHUNK_SIZE
```

or 

```
python3 make_shards.py N_SHARDS N_CLOSEST_CENTERS
```

or 

```
python3 make_shards.py N_SHARDS 
```

or 

```
python3 make_shards.py
```

With default values 
- N_SHARDS = 40
- N_CLOSEST_CENTERS = 2
- N_DIM = 128
- CHUNK_SIZE = 1000

when not specified

### Create EFANNA Graphs

```
python3 create_efanna_graph.py N_SHARDS K L ITERATIONS S R
```

or 

```
python3 create_efanna_graph.py N_SHARDS
```

or 

```
python3 create_efanna_graph.py
```


With default values 
- N_SHARDS = 40
- K = 200
- L = 200
- ITERATIONS = 10
- S = 10
- R = 100

when not specified

### Create NSG Indexes

```
python3 create_nsg_index.py N_SHARDS L R C
```

or 

```
python3 create_nsg_index.py N_SHARDS
```

or 

```
python3 create_nsg_index.py
```


With default values 
- N_SHARDS = 40
- L = 40
- R = 50
- C = 500

when not specified
