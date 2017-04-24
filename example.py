from fasttext import FastVector

fr_dictionary = FastVector(vector_file='wiki.fr.vec')
ru_dictionary = FastVector(vector_file='wiki.ru.vec')

fr_vector = fr_dictionary["chat"]
ru_vector = ru_dictionary["кот"]
print(FastVector.cosine_similarity(fr_vector, ru_vector))

fr_dictionary.apply_transform('alignment_matrices/fr.txt')
ru_dictionary.apply_transform('alignment_matrices/ru.txt')

print(FastVector.cosine_similarity(fr_dictionary["chat"], ru_dictionary["кот"]))