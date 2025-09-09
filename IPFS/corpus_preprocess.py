'''
Seeds of Truth Dataset Processing and Storage

Should preprocess data for LLM training and IPFS storage, plus complete indexing of stored IPFS data. Start with gz of text files and output gz LLM input file and stored/indexed dataset.

1. Intake gz file from Pinata or local file system

2. Chunk text into chunks - maybe 480 tokens/words with an overlap of 40 and smaller chunks padded with surrounding text for uniformity if chunk is end of page/file?
- Associate Source info and other metadata with each chunk
- Create final LLM input file from chunks
- Generate TF-IDF vectors for corpus and store vocab + IDF
- Compute vector space centroids for clustering and search routing
- Turn each chunk into a json array inc at minimum Source and text

3. Use k means to create 20-50 

4. Store each chunk on IPFS, using returned cids + vectors to build indexes and index manifests, inc primary index plus one index per centroid
- Store TF-IDF vocabulary/IDF + manifest info for recomputing values as dataset grows
- Register index manifests on Hive blockchain
- Generate master index of source/question/cid that isn't publicly exposed
- Compute BGE embeddings on the fly for closest n TF-IDF search result values 

'''

