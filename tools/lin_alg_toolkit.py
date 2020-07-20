# lin_alg_toolkit.py



def read_matrix(filename):
    """
    Reads a .txt-file containing whitespace-separated numeric values. Each line 
    of the text file will represent a row in the returned matrix. Matrix is returned
    as a list of lists, where each nested list corresponds to a row of the input data.
    """
    infile = open(filename,'r')
    return [list(map(float,row.split())) for row in infile.readlines()]

# we need a validation function for read_matrix

def tidy_up(matrix):
    """
    Utility-function to tidy up the contents of a matrix by rounding floats to integers
    where possible or to a maximum of three decimal spaces if value is a floating point.
    """
    out = []
    for row in matrix:
        tmp = []
        for value in row:
            if round(value) == round(value,8):
                tmp.append(round(value))
            else:
                tmp.append(round(value,3))
        out.append(tmp)
    return out