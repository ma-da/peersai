from config import *
import corpus_indexing
import calculate_kmeans

def main():
    combined_df = corpus_indexing.do_corpus_indexing()

    combined_df = corpus_indexing.process_trineday_corpus_and_generate_mapping(combined_df)

    vectorizer, X, inv_vocab = corpus_indexing.do_tf_idf_vectorizer_build()

    groups, centroids = calculate_kmeans.calculate_kmeans(X)

    # Mark – Is passing combined df right? N - it looks right if combined_df at this stage is the same as what I was passing to k means in the original script
    calculate_kmeans.build_index_shards(X, vectorizer, groups, centroids, combined_df)

    corpus_indexing.do_local_tf_idf_search_test()

if __name__ == "__main__":
    main()
