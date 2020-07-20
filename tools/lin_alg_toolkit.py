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
    Returns the "tidy" matrix.
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


def scalar_multiply(matrix,scalar):
    """
    Returns input parameter matrix  multiplied by a scalar
    """
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] *= scalar
    return matrix



def dot_product(matrix_A,matrix_B):
    """
    Returns the product of matrix_A multiplied by matrix_B in this order. 
    Returns nothing if product is not defined - that is matrix_A not having
    as many columns as matrix_B has rows.
    """
    cols_A = len(matrix_A[0])
    rows_B = len(matrix_B)
    cols_B = len(matrix_B[0])

    if cols_A != rows_B:
        return

    B_columns = []   # maybe write a transpose cols function
    for i in range(cols_B):
        tmp = []
        for j in range(rows_B):
             tmp.append(matrix_B[j][i])
        B_columns.append(tmp)

    out = []
    for row in matrix_A:
        tmp = []
        for i in range(cols_B):
            val = 0
            col_to_multiply = B_columns[i]
            for j in range(len(col_to_multiply)):
                val += row[j] * col_to_multiply[j]
            tmp.append(val)
        out.append(tmp)   
    return out



def create_identity_matrix(order):
    out = []
    for i in range(order):
        tmp = []
        for j in range(order):
            if i == j:
                tmp.append(1)
            else:
                tmp.append(0)
        out.append(tmp)
    return out