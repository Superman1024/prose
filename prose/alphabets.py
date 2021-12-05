"""
Copyright (C) Tristan Bepler - All Rights Reserved
Author: Tristan Bepler <tbepler@gmail.com>
"""

from __future__ import print_function, division

import numpy as np

class Alphabet:
    def __init__(self, chars, encoding=None, mask=False, missing=255):
        self.chars = np.frombuffer(chars, dtype=np.uint8)  # load as ASCII code
        self.encoding = np.zeros(256, dtype=np.uint8) + missing  # len=256, value equal to `missing`

        if encoding is None:
            self.encoding[self.chars] = np.arange(len(self.chars))
            self.size = len(self.chars)
        else:
            self.encoding[self.chars] = encoding
            self.size = encoding.max() + 1  # start with 0, size = max + 1
        # replace correspond place (it depends on the ASCII code) with 0 1 2 3 ...
        # The result means, among 256 numbers, some have index 0 1 2 ... , the other is 255

        self.mask = mask
        if mask:
            self.size -= 1  # predict the masked position using the rest (size - 1)

    def __len__(self):
        return self.size

    def __getitem__(self, i):  # return a `char` correspond to the number `i`, which is an ASCII code
        return chr(self.chars[i])

    def encode(self, x):  # input a byte string, only one! return 0 1 2 ... 19 20 for this byte
        """ encode a byte string into alphabet indices """
        x = np.frombuffer(x, dtype=np.uint8)  # load as ASCII code
        return self.encoding[x]

    def decode(self, x):  # input a number as index among, 0 1 2 ... 19 20, output the character
        """ decode index array, x, to byte string of this alphabet """
        string = self.chars[x]  # convert to ASCII
        return string.tobytes()  # eg: b'A' for string = 65

    def unpack(self, h, k):
        """ unpack integer h into array of this alphabet with length k """
        n = self.size
        kmer = np.zeros(k, dtype=np.uint8)
        for i in reversed(range(k)):
            c = h % n
            kmer[i] = c
            h = h // n
        return kmer

    def get_kmer(self, h, k):
        """ retrieve byte string of length k decoded from integer h """
        kmer = self.unpack(h, k)
        return self.decode(kmer)
        
DNA = Alphabet(b'ACGT')
        
class Uniprot21(Alphabet):
    def __init__(self, mask=False):
        chars = alphabet = b'ARNDCQEGHILKMFPSTWYVXOUBZ'
        encoding = np.arange(len(chars))
        encoding[21:] = [11,4,20,20] # encode 'OUBZ' as synonyms
        # After this step, the unique char only have 21
        super(Uniprot21, self).__init__(
            chars,
            encoding=encoding,  # use special encoding, encode 'OUBZ' as synonyms
            mask=mask,
            missing=20  # If there exist unexpected char, consider it as `X`
        )

class SDM12(Alphabet):
    """
    A D KER N TSQ YF LIVM C W H G P

    See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2732308/#B33
    "Reduced amino acid alphabets exhibit an improved sensitivity and selectivity in fold assignment"
    Peterson et al. 2009. Bioinformatics.
    """
    def __init__(self, mask=False):
        chars = alphabet = b'ADKNTYLCWHGPXERSQFIVMOUBZ'
        groups = [b'A',b'D',b'KERO',b'N',b'TSQ',b'YF',b'LIVM',b'CU',b'W',b'H',b'G',b'P',b'XBZ']
        groups = {c:i for i in range(len(groups)) for c in groups[i]}
        encoding = np.array([groups[c] for c in chars])
        super(SDM12, self).__init__(chars, encoding=encoding, mask=mask)

SecStr8 = Alphabet(b'HBEGITS ')



