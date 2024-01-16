from parameters.general_paramters import  motifs_full_name

xors_6bit_network_dims = [6, 6, 4, 4, 2, 2]

xor_structural_features_vec_length = 96
xor_structural_features_vec_length_with_motifs = 99

XOR_EDGE_MAPPING = [
    '((0, 0), (0, 0))',
    '((0, 0), (0, 1))',
    '((0, 0), (0, 2))',
    '((0, 0), (0, 3))',
    '((0, 0), (0, 4))',
    '((0, 0), (0, 5))',
    '((0, 0), (1, 0))',
    '((0, 0), (1, 1))',
    '((0, 0), (1, 2))',
    '((0, 0), (1, 3))',
    '((0, 0), (1, 4))',
    '((0, 0), (1, 5))',
    '((0, 0), (2, 0))',
    '((0, 0), (2, 1))',
    '((0, 0), (2, 2))',
    '((0, 0), (2, 3))',
    '((0, 0), (3, 0))',
    '((0, 0), (3, 1))',
    '((0, 0), (3, 2))',
    '((0, 0), (3, 3))',
    '((0, 0), (4, 0))',
    '((0, 0), (4, 1))',
    '((0, 0), (5, 0))',
    '((0, 0), (5, 1))',
    '((0, 1), (0, 0))',
    '((0, 1), (0, 1))',
    '((0, 1), (0, 2))',
    '((0, 1), (0, 3))',
    '((0, 1), (0, 4))',
    '((0, 1), (0, 5))',
    '((0, 1), (1, 0))',
    '((0, 1), (1, 1))',
    '((0, 1), (1, 2))',
    '((0, 1), (1, 3))',
    '((0, 1), (1, 4))',
    '((0, 1), (1, 5))',
    '((0, 1), (2, 0))',
    '((0, 1), (2, 1))',
    '((0, 1), (2, 2))',
    '((0, 1), (2, 3))',
    '((0, 1), (3, 0))',
    '((0, 1), (3, 1))',
    '((0, 1), (3, 2))',
    '((0, 1), (3, 3))',
    '((0, 1), (4, 0))',
    '((0, 1), (4, 1))',
    '((0, 1), (5, 0))',
    '((0, 1), (5, 1))',
    '((0, 2), (0, 0))',
    '((0, 2), (0, 1))',
    '((0, 2), (0, 2))',
    '((0, 2), (0, 3))',
    '((0, 2), (0, 4))',
    '((0, 2), (0, 5))',
    '((0, 2), (1, 0))',
    '((0, 2), (1, 1))',
    '((0, 2), (1, 2))',
    '((0, 2), (1, 3))',
    '((0, 2), (1, 4))',
    '((0, 2), (1, 5))',
    '((0, 2), (2, 0))',
    '((0, 2), (2, 1))',
    '((0, 2), (2, 2))',
    '((0, 2), (2, 3))',
    '((0, 2), (3, 0))',
    '((0, 2), (3, 1))',
    '((0, 2), (3, 2))',
    '((0, 2), (3, 3))',
    '((0, 2), (4, 0))',
    '((0, 2), (4, 1))',
    '((0, 2), (5, 0))',
    '((0, 2), (5, 1))',
    '((0, 3), (0, 0))',
    '((0, 3), (0, 1))',
    '((0, 3), (0, 2))',
    '((0, 3), (0, 3))',
    '((0, 3), (0, 4))',
    '((0, 3), (0, 5))',
    '((0, 3), (1, 0))',
    '((0, 3), (1, 1))',
    '((0, 3), (1, 2))',
    '((0, 3), (1, 3))',
    '((0, 3), (1, 4))',
    '((0, 3), (1, 5))',
    '((0, 3), (2, 0))',
    '((0, 3), (2, 1))',
    '((0, 3), (2, 2))',
    '((0, 3), (2, 3))',
    '((0, 3), (3, 0))',
    '((0, 3), (3, 1))',
    '((0, 3), (3, 2))',
    '((0, 3), (3, 3))',
    '((0, 3), (4, 0))',
    '((0, 3), (4, 1))',
    '((0, 3), (5, 0))',
    '((0, 3), (5, 1))',
    '((0, 4), (0, 0))',
    '((0, 4), (0, 1))',
    '((0, 4), (0, 2))',
    '((0, 4), (0, 3))',
    '((0, 4), (0, 4))',
    '((0, 4), (0, 5))',
    '((0, 4), (1, 0))',
    '((0, 4), (1, 1))',
    '((0, 4), (1, 2))',
    '((0, 4), (1, 3))',
    '((0, 4), (1, 4))',
    '((0, 4), (1, 5))',
    '((0, 4), (2, 0))',
    '((0, 4), (2, 1))',
    '((0, 4), (2, 2))',
    '((0, 4), (2, 3))',
    '((0, 4), (3, 0))',
    '((0, 4), (3, 1))',
    '((0, 4), (3, 2))',
    '((0, 4), (3, 3))',
    '((0, 4), (4, 0))',
    '((0, 4), (4, 1))',
    '((0, 4), (5, 0))',
    '((0, 4), (5, 1))',
    '((0, 5), (0, 0))',
    '((0, 5), (0, 1))',
    '((0, 5), (0, 2))',
    '((0, 5), (0, 3))',
    '((0, 5), (0, 4))',
    '((0, 5), (0, 5))',
    '((0, 5), (1, 0))',
    '((0, 5), (1, 1))',
    '((0, 5), (1, 2))',
    '((0, 5), (1, 3))',
    '((0, 5), (1, 4))',
    '((0, 5), (1, 5))',
    '((0, 5), (2, 0))',
    '((0, 5), (2, 1))',
    '((0, 5), (2, 2))',
    '((0, 5), (2, 3))',
    '((0, 5), (3, 0))',
    '((0, 5), (3, 1))',
    '((0, 5), (3, 2))',
    '((0, 5), (3, 3))',
    '((0, 5), (4, 0))',
    '((0, 5), (4, 1))',
    '((0, 5), (5, 0))',
    '((0, 5), (5, 1))',
    '((1, 0), (0, 0))',
    '((1, 0), (0, 1))',
    '((1, 0), (0, 2))',
    '((1, 0), (0, 3))',
    '((1, 0), (0, 4))',
    '((1, 0), (0, 5))',
    '((1, 0), (1, 0))',
    '((1, 0), (1, 1))',
    '((1, 0), (1, 2))',
    '((1, 0), (1, 3))',
    '((1, 0), (1, 4))',
    '((1, 0), (1, 5))',
    '((1, 0), (2, 0))',
    '((1, 0), (2, 1))',
    '((1, 0), (2, 2))',
    '((1, 0), (2, 3))',
    '((1, 0), (3, 0))',
    '((1, 0), (3, 1))',
    '((1, 0), (3, 2))',
    '((1, 0), (3, 3))',
    '((1, 0), (4, 0))',
    '((1, 0), (4, 1))',
    '((1, 0), (5, 0))',
    '((1, 0), (5, 1))',
    '((1, 1), (0, 0))',
    '((1, 1), (0, 1))',
    '((1, 1), (0, 2))',
    '((1, 1), (0, 3))',
    '((1, 1), (0, 4))',
    '((1, 1), (0, 5))',
    '((1, 1), (1, 0))',
    '((1, 1), (1, 1))',
    '((1, 1), (1, 2))',
    '((1, 1), (1, 3))',
    '((1, 1), (1, 4))',
    '((1, 1), (1, 5))',
    '((1, 1), (2, 0))',
    '((1, 1), (2, 1))',
    '((1, 1), (2, 2))',
    '((1, 1), (2, 3))',
    '((1, 1), (3, 0))',
    '((1, 1), (3, 1))',
    '((1, 1), (3, 2))',
    '((1, 1), (3, 3))',
    '((1, 1), (4, 0))',
    '((1, 1), (4, 1))',
    '((1, 1), (5, 0))',
    '((1, 1), (5, 1))',
    '((1, 2), (0, 0))',
    '((1, 2), (0, 1))',
    '((1, 2), (0, 2))',
    '((1, 2), (0, 3))',
    '((1, 2), (0, 4))',
    '((1, 2), (0, 5))',
    '((1, 2), (1, 0))',
    '((1, 2), (1, 1))',
    '((1, 2), (1, 2))',
    '((1, 2), (1, 3))',
    '((1, 2), (1, 4))',
    '((1, 2), (1, 5))',
    '((1, 2), (2, 0))',
    '((1, 2), (2, 1))',
    '((1, 2), (2, 2))',
    '((1, 2), (2, 3))',
    '((1, 2), (3, 0))',
    '((1, 2), (3, 1))',
    '((1, 2), (3, 2))',
    '((1, 2), (3, 3))',
    '((1, 2), (4, 0))',
    '((1, 2), (4, 1))',
    '((1, 2), (5, 0))',
    '((1, 2), (5, 1))',
    '((1, 3), (0, 0))',
    '((1, 3), (0, 1))',
    '((1, 3), (0, 2))',
    '((1, 3), (0, 3))',
    '((1, 3), (0, 4))',
    '((1, 3), (0, 5))',
    '((1, 3), (1, 0))',
    '((1, 3), (1, 1))',
    '((1, 3), (1, 2))',
    '((1, 3), (1, 3))',
    '((1, 3), (1, 4))',
    '((1, 3), (1, 5))',
    '((1, 3), (2, 0))',
    '((1, 3), (2, 1))',
    '((1, 3), (2, 2))',
    '((1, 3), (2, 3))',
    '((1, 3), (3, 0))',
    '((1, 3), (3, 1))',
    '((1, 3), (3, 2))',
    '((1, 3), (3, 3))',
    '((1, 3), (4, 0))',
    '((1, 3), (4, 1))',
    '((1, 3), (5, 0))',
    '((1, 3), (5, 1))',
    '((1, 4), (0, 0))',
    '((1, 4), (0, 1))',
    '((1, 4), (0, 2))',
    '((1, 4), (0, 3))',
    '((1, 4), (0, 4))',
    '((1, 4), (0, 5))',
    '((1, 4), (1, 0))',
    '((1, 4), (1, 1))',
    '((1, 4), (1, 2))',
    '((1, 4), (1, 3))',
    '((1, 4), (1, 4))',
    '((1, 4), (1, 5))',
    '((1, 4), (2, 0))',
    '((1, 4), (2, 1))',
    '((1, 4), (2, 2))',
    '((1, 4), (2, 3))',
    '((1, 4), (3, 0))',
    '((1, 4), (3, 1))',
    '((1, 4), (3, 2))',
    '((1, 4), (3, 3))',
    '((1, 4), (4, 0))',
    '((1, 4), (4, 1))',
    '((1, 4), (5, 0))',
    '((1, 4), (5, 1))',
    '((1, 5), (0, 0))',
    '((1, 5), (0, 1))',
    '((1, 5), (0, 2))',
    '((1, 5), (0, 3))',
    '((1, 5), (0, 4))',
    '((1, 5), (0, 5))',
    '((1, 5), (1, 0))',
    '((1, 5), (1, 1))',
    '((1, 5), (1, 2))',
    '((1, 5), (1, 3))',
    '((1, 5), (1, 4))',
    '((1, 5), (1, 5))',
    '((1, 5), (2, 0))',
    '((1, 5), (2, 1))',
    '((1, 5), (2, 2))',
    '((1, 5), (2, 3))',
    '((1, 5), (3, 0))',
    '((1, 5), (3, 1))',
    '((1, 5), (3, 2))',
    '((1, 5), (3, 3))',
    '((1, 5), (4, 0))',
    '((1, 5), (4, 1))',
    '((1, 5), (5, 0))',
    '((1, 5), (5, 1))',
    '((2, 0), (0, 0))',
    '((2, 0), (0, 1))',
    '((2, 0), (0, 2))',
    '((2, 0), (0, 3))',
    '((2, 0), (0, 4))',
    '((2, 0), (0, 5))',
    '((2, 0), (1, 0))',
    '((2, 0), (1, 1))',
    '((2, 0), (1, 2))',
    '((2, 0), (1, 3))',
    '((2, 0), (1, 4))',
    '((2, 0), (1, 5))',
    '((2, 0), (2, 0))',
    '((2, 0), (2, 1))',
    '((2, 0), (2, 2))',
    '((2, 0), (2, 3))',
    '((2, 0), (3, 0))',
    '((2, 0), (3, 1))',
    '((2, 0), (3, 2))',
    '((2, 0), (3, 3))',
    '((2, 0), (4, 0))',
    '((2, 0), (4, 1))',
    '((2, 0), (5, 0))',
    '((2, 0), (5, 1))',
    '((2, 1), (0, 0))',
    '((2, 1), (0, 1))',
    '((2, 1), (0, 2))',
    '((2, 1), (0, 3))',
    '((2, 1), (0, 4))',
    '((2, 1), (0, 5))',
    '((2, 1), (1, 0))',
    '((2, 1), (1, 1))',
    '((2, 1), (1, 2))',
    '((2, 1), (1, 3))',
    '((2, 1), (1, 4))',
    '((2, 1), (1, 5))',
    '((2, 1), (2, 0))',
    '((2, 1), (2, 1))',
    '((2, 1), (2, 2))',
    '((2, 1), (2, 3))',
    '((2, 1), (3, 0))',
    '((2, 1), (3, 1))',
    '((2, 1), (3, 2))',
    '((2, 1), (3, 3))',
    '((2, 1), (4, 0))',
    '((2, 1), (4, 1))',
    '((2, 1), (5, 0))',
    '((2, 1), (5, 1))',
    '((2, 2), (0, 0))',
    '((2, 2), (0, 1))',
    '((2, 2), (0, 2))',
    '((2, 2), (0, 3))',
    '((2, 2), (0, 4))',
    '((2, 2), (0, 5))',
    '((2, 2), (1, 0))',
    '((2, 2), (1, 1))',
    '((2, 2), (1, 2))',
    '((2, 2), (1, 3))',
    '((2, 2), (1, 4))',
    '((2, 2), (1, 5))',
    '((2, 2), (2, 0))',
    '((2, 2), (2, 1))',
    '((2, 2), (2, 2))',
    '((2, 2), (2, 3))',
    '((2, 2), (3, 0))',
    '((2, 2), (3, 1))',
    '((2, 2), (3, 2))',
    '((2, 2), (3, 3))',
    '((2, 2), (4, 0))',
    '((2, 2), (4, 1))',
    '((2, 2), (5, 0))',
    '((2, 2), (5, 1))',
    '((2, 3), (0, 0))',
    '((2, 3), (0, 1))',
    '((2, 3), (0, 2))',
    '((2, 3), (0, 3))',
    '((2, 3), (0, 4))',
    '((2, 3), (0, 5))',
    '((2, 3), (1, 0))',
    '((2, 3), (1, 1))',
    '((2, 3), (1, 2))',
    '((2, 3), (1, 3))',
    '((2, 3), (1, 4))',
    '((2, 3), (1, 5))',
    '((2, 3), (2, 0))',
    '((2, 3), (2, 1))',
    '((2, 3), (2, 2))',
    '((2, 3), (2, 3))',
    '((2, 3), (3, 0))',
    '((2, 3), (3, 1))',
    '((2, 3), (3, 2))',
    '((2, 3), (3, 3))',
    '((2, 3), (4, 0))',
    '((2, 3), (4, 1))',
    '((2, 3), (5, 0))',
    '((2, 3), (5, 1))',
    '((3, 0), (0, 0))',
    '((3, 0), (0, 1))',
    '((3, 0), (0, 2))',
    '((3, 0), (0, 3))',
    '((3, 0), (0, 4))',
    '((3, 0), (0, 5))',
    '((3, 0), (1, 0))',
    '((3, 0), (1, 1))',
    '((3, 0), (1, 2))',
    '((3, 0), (1, 3))',
    '((3, 0), (1, 4))',
    '((3, 0), (1, 5))',
    '((3, 0), (2, 0))',
    '((3, 0), (2, 1))',
    '((3, 0), (2, 2))',
    '((3, 0), (2, 3))',
    '((3, 0), (3, 0))',
    '((3, 0), (3, 1))',
    '((3, 0), (3, 2))',
    '((3, 0), (3, 3))',
    '((3, 0), (4, 0))',
    '((3, 0), (4, 1))',
    '((3, 0), (5, 0))',
    '((3, 0), (5, 1))',
    '((3, 1), (0, 0))',
    '((3, 1), (0, 1))',
    '((3, 1), (0, 2))',
    '((3, 1), (0, 3))',
    '((3, 1), (0, 4))',
    '((3, 1), (0, 5))',
    '((3, 1), (1, 0))',
    '((3, 1), (1, 1))',
    '((3, 1), (1, 2))',
    '((3, 1), (1, 3))',
    '((3, 1), (1, 4))',
    '((3, 1), (1, 5))',
    '((3, 1), (2, 0))',
    '((3, 1), (2, 1))',
    '((3, 1), (2, 2))',
    '((3, 1), (2, 3))',
    '((3, 1), (3, 0))',
    '((3, 1), (3, 1))',
    '((3, 1), (3, 2))',
    '((3, 1), (3, 3))',
    '((3, 1), (4, 0))',
    '((3, 1), (4, 1))',
    '((3, 1), (5, 0))',
    '((3, 1), (5, 1))',
    '((3, 2), (0, 0))',
    '((3, 2), (0, 1))',
    '((3, 2), (0, 2))',
    '((3, 2), (0, 3))',
    '((3, 2), (0, 4))',
    '((3, 2), (0, 5))',
    '((3, 2), (1, 0))',
    '((3, 2), (1, 1))',
    '((3, 2), (1, 2))',
    '((3, 2), (1, 3))',
    '((3, 2), (1, 4))',
    '((3, 2), (1, 5))',
    '((3, 2), (2, 0))',
    '((3, 2), (2, 1))',
    '((3, 2), (2, 2))',
    '((3, 2), (2, 3))',
    '((3, 2), (3, 0))',
    '((3, 2), (3, 1))',
    '((3, 2), (3, 2))',
    '((3, 2), (3, 3))',
    '((3, 2), (4, 0))',
    '((3, 2), (4, 1))',
    '((3, 2), (5, 0))',
    '((3, 2), (5, 1))',
    '((3, 3), (0, 0))',
    '((3, 3), (0, 1))',
    '((3, 3), (0, 2))',
    '((3, 3), (0, 3))',
    '((3, 3), (0, 4))',
    '((3, 3), (0, 5))',
    '((3, 3), (1, 0))',
    '((3, 3), (1, 1))',
    '((3, 3), (1, 2))',
    '((3, 3), (1, 3))',
    '((3, 3), (1, 4))',
    '((3, 3), (1, 5))',
    '((3, 3), (2, 0))',
    '((3, 3), (2, 1))',
    '((3, 3), (2, 2))',
    '((3, 3), (2, 3))',
    '((3, 3), (3, 0))',
    '((3, 3), (3, 1))',
    '((3, 3), (3, 2))',
    '((3, 3), (3, 3))',
    '((3, 3), (4, 0))',
    '((3, 3), (4, 1))',
    '((3, 3), (5, 0))',
    '((3, 3), (5, 1))',
    '((4, 0), (0, 0))',
    '((4, 0), (0, 1))',
    '((4, 0), (0, 2))',
    '((4, 0), (0, 3))',
    '((4, 0), (0, 4))',
    '((4, 0), (0, 5))',
    '((4, 0), (1, 0))',
    '((4, 0), (1, 1))',
    '((4, 0), (1, 2))',
    '((4, 0), (1, 3))',
    '((4, 0), (1, 4))',
    '((4, 0), (1, 5))',
    '((4, 0), (2, 0))',
    '((4, 0), (2, 1))',
    '((4, 0), (2, 2))',
    '((4, 0), (2, 3))',
    '((4, 0), (3, 0))',
    '((4, 0), (3, 1))',
    '((4, 0), (3, 2))',
    '((4, 0), (3, 3))',
    '((4, 0), (4, 0))',
    '((4, 0), (4, 1))',
    '((4, 0), (5, 0))',
    '((4, 0), (5, 1))',
    '((4, 1), (0, 0))',
    '((4, 1), (0, 1))',
    '((4, 1), (0, 2))',
    '((4, 1), (0, 3))',
    '((4, 1), (0, 4))',
    '((4, 1), (0, 5))',
    '((4, 1), (1, 0))',
    '((4, 1), (1, 1))',
    '((4, 1), (1, 2))',
    '((4, 1), (1, 3))',
    '((4, 1), (1, 4))',
    '((4, 1), (1, 5))',
    '((4, 1), (2, 0))',
    '((4, 1), (2, 1))',
    '((4, 1), (2, 2))',
    '((4, 1), (2, 3))',
    '((4, 1), (3, 0))',
    '((4, 1), (3, 1))',
    '((4, 1), (3, 2))',
    '((4, 1), (3, 3))',
    '((4, 1), (4, 0))',
    '((4, 1), (4, 1))',
    '((4, 1), (5, 0))',
    '((4, 1), (5, 1))',
    '((5, 0), (0, 0))',
    '((5, 0), (0, 1))',
    '((5, 0), (0, 2))',
    '((5, 0), (0, 3))',
    '((5, 0), (0, 4))',
    '((5, 0), (0, 5))',
    '((5, 0), (1, 0))',
    '((5, 0), (1, 1))',
    '((5, 0), (1, 2))',
    '((5, 0), (1, 3))',
    '((5, 0), (1, 4))',
    '((5, 0), (1, 5))',
    '((5, 0), (2, 0))',
    '((5, 0), (2, 1))',
    '((5, 0), (2, 2))',
    '((5, 0), (2, 3))',
    '((5, 0), (3, 0))',
    '((5, 0), (3, 1))',
    '((5, 0), (3, 2))',
    '((5, 0), (3, 3))',
    '((5, 0), (4, 0))',
    '((5, 0), (4, 1))',
    '((5, 0), (5, 0))',
    '((5, 0), (5, 1))',
    '((5, 1), (0, 0))',
    '((5, 1), (0, 1))',
    '((5, 1), (0, 2))',
    '((5, 1), (0, 3))',
    '((5, 1), (0, 4))',
    '((5, 1), (0, 5))',
    '((5, 1), (1, 0))',
    '((5, 1), (1, 1))',
    '((5, 1), (1, 2))',
    '((5, 1), (1, 3))',
    '((5, 1), (1, 4))',
    '((5, 1), (1, 5))',
    '((5, 1), (2, 0))',
    '((5, 1), (2, 1))',
    '((5, 1), (2, 2))',
    '((5, 1), (2, 3))',
    '((5, 1), (3, 0))',
    '((5, 1), (3, 1))',
    '((5, 1), (3, 2))',
    '((5, 1), (3, 3))',
    '((5, 1), (4, 0))',
    '((5, 1), (4, 1))',
    '((5, 1), (5, 0))',
    '((5, 1), (5, 1))']

xor_structural_features_full_name_vec = [
    'modularity',
    'entropy',
    'normed_entropy',
    'connectivity_ratio',
    'num_connections',
    'max_possible_connections',
    'out_connections_per_layer_(0, 0) ',
    'out_connections_per_layer_(0, 1) ',
    'out_connections_per_layer_(0, 2) ',
    'out_connections_per_layer_(0, 3) ',
    'out_connections_per_layer_(0, 4) ',
    'out_connections_per_layer_(0, 5) ',
    'out_connections_per_layer_(1, 0) ',
    'out_connections_per_layer_(1, 1) ',
    'out_connections_per_layer_(1, 2) ',
    'out_connections_per_layer_(1, 3) ',
    'out_connections_per_layer_(1, 4) ',
    'out_connections_per_layer_(1, 5) ',
    'out_connections_per_layer_(2, 0) ',
    'out_connections_per_layer_(2, 1) ',
    'out_connections_per_layer_(2, 2) ',
    'out_connections_per_layer_(2, 3) ',
    'out_connections_per_layer_(3, 0) ',
    'out_connections_per_layer_(3, 1) ',
    'out_connections_per_layer_(3, 2) ',
    'out_connections_per_layer_(3, 3) ',
    'out_connections_per_layer_(4, 0) ',
    'out_connections_per_layer_(4, 1) ',
    'out_connections_per_layer_(5, 0) ',
    'out_connections_per_layer_(5, 1) ',
    'in_connections_per_layer_(0, 0) ',
    'in_connections_per_layer_(0, 1) ',
    'in_connections_per_layer_(0, 2) ',
    'in_connections_per_layer_(0, 3) ',
    'in_connections_per_layer_(0, 4) ',
    'in_connections_per_layer_(0, 5) ',
    'in_connections_per_layer_(1, 0) ',
    'in_connections_per_layer_(1, 1) ',
    'in_connections_per_layer_(1, 2) ',
    'in_connections_per_layer_(1, 3) ',
    'in_connections_per_layer_(1, 4) ',
    'in_connections_per_layer_(1, 5) ',
    'in_connections_per_layer_(2, 0) ',
    'in_connections_per_layer_(2, 1) ',
    'in_connections_per_layer_(2, 2) ',
    'in_connections_per_layer_(2, 3) ',
    'in_connections_per_layer_(3, 0) ',
    'in_connections_per_layer_(3, 1) ',
    'in_connections_per_layer_(3, 2) ',
    'in_connections_per_layer_(3, 3) ',
    'in_connections_per_layer_(4, 0) ',
    'in_connections_per_layer_(4, 1) ',
    'in_connections_per_layer_(5, 0) ',
    'in_connections_per_layer_(5, 1) ',
    'total_connectivity_ratio_between_layers_0',
    'total_connectivity_ratio_between_layers_1',
    'total_connectivity_ratio_between_layers_2',
    'total_connectivity_ratio_between_layers_3',
    'total_connectivity_ratio_between_layers_4',
    'max_connectivity_between_layers_per_layer_0',
    'max_connectivity_between_layers_per_layer_1',
    'max_connectivity_between_layers_per_layer_2',
    'max_connectivity_between_layers_per_layer_3',
    'max_connectivity_between_layers_per_layer_4',
    'layer_connectivity_rank_0',
    'layer_connectivity_rank_1',
    'layer_connectivity_rank_2',
    'layer_connectivity_rank_3',
    'layer_connectivity_rank_4',
    'distances_between_input_neuron_0',
    'distances_between_input_neuron_1',
    'distances_between_input_neuron_2',
    'distances_between_input_neuron_3',
    'distances_between_input_neuron_4',
    'distances_between_input_neuron_5',
    'distances_between_input_neuron_6',
    'distances_between_input_neuron_7',
    'distances_between_input_neuron_8',
    'distances_between_input_neuron_9',
    'distances_between_input_neuron_10',
    'distances_between_input_neuron_11',
    'distances_between_input_neuron_12',
    'distances_between_input_neuron_13',
    'distances_between_input_neuron_14',
    'num_paths_to_output_per_input_neuron_(0, 0)',
    'num_paths_to_output_per_input_neuron_(0, 1)',
    'num_paths_to_output_per_input_neuron_(0, 2)',
    'num_paths_to_output_per_input_neuron_(0, 3)',
    'num_paths_to_output_per_input_neuron_(0, 4)',
    'num_paths_to_output_per_input_neuron_(0, 5)',
    'num_involved_neurons_in_paths_per_input_neuron_(0, 0)',
    'num_involved_neurons_in_paths_per_input_neuron_(0, 1)',
    'num_involved_neurons_in_paths_per_input_neuron_(0, 2)',
    'num_involved_neurons_in_paths_per_input_neuron_(0, 3)',
    'num_involved_neurons_in_paths_per_input_neuron_(0, 4)',
    'num_involved_neurons_in_paths_per_input_neuron_(0, 5)',
]
xor_structural_features_name_vec = [
    'modularity',
    'entropy',
    'normed_entropy',
    'connectivity_ratio',
    'num_connections',
    'max_possible_connections',
    'out_connections_per_layer_0',
    'out_connections_per_layer_0',
    'out_connections_per_layer_0',
    'out_connections_per_layer_0',
    'out_connections_per_layer_0',
    'out_connections_per_layer_0',
    'out_connections_per_layer_1',
    'out_connections_per_layer_1',
    'out_connections_per_layer_1',
    'out_connections_per_layer_1',
    'out_connections_per_layer_1',
    'out_connections_per_layer_1',
    'out_connections_per_layer_2',
    'out_connections_per_layer_2',
    'out_connections_per_layer_2',
    'out_connections_per_layer_2',
    'out_connections_per_layer_3',
    'out_connections_per_layer_3',
    'out_connections_per_layer_3',
    'out_connections_per_layer_3',
    'out_connections_per_layer_4',
    'out_connections_per_layer_4',
    'out_connections_per_layer_5',
    'out_connections_per_layer_5',
    'in_connections_per_layer_0',
    'in_connections_per_layer_0',
    'in_connections_per_layer_0',
    'in_connections_per_layer_0',
    'in_connections_per_layer_0',
    'in_connections_per_layer_0',
    'in_connections_per_layer_1',
    'in_connections_per_layer_1',
    'in_connections_per_layer_1',
    'in_connections_per_layer_1',
    'in_connections_per_layer_1',
    'in_connections_per_layer_1',
    'in_connections_per_layer_2',
    'in_connections_per_layer_2',
    'in_connections_per_layer_2',
    'in_connections_per_layer_2',
    'in_connections_per_layer_3',
    'in_connections_per_layer_3',
    'in_connections_per_layer_3',
    'in_connections_per_layer_3',
    'in_connections_per_layer_4',
    'in_connections_per_layer_4',
    'in_connections_per_layer_5',
    'in_connections_per_layer_5',
    'total_connectivity_ratio_between_layers_0',
    'total_connectivity_ratio_between_layers_1',
    'total_connectivity_ratio_between_layers_2',
    'total_connectivity_ratio_between_layers_3',
    'total_connectivity_ratio_between_layers_4',
    'max_connectivity_between_layers_per_layer_0',
    'max_connectivity_between_layers_per_layer_1',
    'max_connectivity_between_layers_per_layer_2',
    'max_connectivity_between_layers_per_layer_3',
    'max_connectivity_between_layers_per_layer_4',
    'layer_connectivity_rank_0',
    'layer_connectivity_rank_1',
    'layer_connectivity_rank_2',
    'layer_connectivity_rank_3',
    'layer_connectivity_rank_4',
    'distances_between_input_neuron',
    'distances_between_input_neuron',
    'distances_between_input_neuron',
    'distances_between_input_neuron',
    'distances_between_input_neuron',
    'distances_between_input_neuron',
    'distances_between_input_neuron',
    'distances_between_input_neuron',
    'distances_between_input_neuron',
    'distances_between_input_neuron',
    'distances_between_input_neuron',
    'distances_between_input_neuron',
    'distances_between_input_neuron',
    'distances_between_input_neuron',
    'distances_between_input_neuron',
    'num_paths_to_output_per_input_neuron',
    'num_paths_to_output_per_input_neuron',
    'num_paths_to_output_per_input_neuron',
    'num_paths_to_output_per_input_neuron',
    'num_paths_to_output_per_input_neuron',
    'num_paths_to_output_per_input_neuron',
    'num_involved_neurons_in_paths_per_input_neuron',
    'num_involved_neurons_in_paths_per_input_neuron',
    'num_involved_neurons_in_paths_per_input_neuron',
    'num_involved_neurons_in_paths_per_input_neuron',
    'num_involved_neurons_in_paths_per_input_neuron',
    'num_involved_neurons_in_paths_per_input_neuron',
]
xor_structural_features_full_name_vec_with_motifs = xor_structural_features_full_name_vec + motifs_full_name
