'''
Seeds of Truth Dataset Processing and Storage

Should preprocess data for LLM training and IPFS storage, plus complete indexing of stored IPFS data. Start with gz of text files and output gz LLM input file and stored/indexed dataset.

1. Intake gz file from Pinata or local file system
2. Chunk text into chunks - maybe 480 tokens max with an overlap of 40 and smaller chunks okay if chunk is end of page/file?
- Add Source and Question info (plus stop token?) to each chunk
- Create LLM input file from annotated chunks
- Generate TF-IDF and BGE vectors for corpus
- Compute vector space centroids for clustering and search routing
- Turn each chunk into a json file inc at minimum Source, Question, and text
3. Store each chunk on IPFS, using returned cids + vectors to build indexes and index manifests for TF-IDF and BGE, inc primary index plus one index per centroid
- Store TF-IDF vocabulary + manifest info for recomputing values as dataset grows
- Register index manifests on Hive blockchain
- Generate master index of source/question/cid that isn't publicly exposed    
'''