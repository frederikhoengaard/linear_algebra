# lin_alg_toolkit.py



def read_matrix(filename):
    """
    Reads a .txt-file containing whitespace-separated numeric values in the following form:
    3 3 -2 11
    2 0 3 -1
    -1 -2 0 -5
    ...
    Where each line of the text file will represent a row in the returned matrix. The matrix 
    is returned as a list of lists, where each nested list corresponds to a row of the input data.
    """
    infile = open(filename,'r')
    return [list(map(float,row.split())) for row in infile.readlines()]

# we need a validation function for read_matrix

def get_size(matrix):
    """
    Returns a list [m,n] where m is the number of rows and n is the number of columns of the input
    matrix.
    """
    return [len(matrix),len(matrix[0])]



def tidy_up(matrix):
    """
    Utility-function to tidy up the contents of a matrix by rounding floats to integers
    where possible or to a maximum of three decimal spaces if value is a floating point.
    Returns the "tidy" matrix.
    """
    tidier_matrix = []
    for row in matrix:
        tmp = []
        for value in row:
            if round(value) == round(value,8):
                tmp.append(round(value))
            else:
                tmp.append(round(value,3))
        tidier_matrix.append(tmp)
    return tidier_matrix



def transpose_matrix(matrix):
    """
    Transposes the input matrix by interchanging the rows and columns. Returns the 
    transposed matrix.
    """    
    m,n = get_size(matrix)
    transposed_matrix = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(matrix[j][i])
        transposed_matrix.append(row)
    return transposed_matrix



def matrix_addition(matrix_A,matrix_B):
    """
    Returns the sum of two equal-sized matrices. If the input parameter matrices are not of equal
    size nothing is returned.
    """
    m_A, n_A = get_size(matrix_A)
    m_B, n_B = get_size(matrix_B)
    
    if m_A != m_B or n_A != n_B:
        return

    matrix_sum = []
    for i in range(m_A):
        row = []
        for j in range(n_A):
            row.append(matrix_A[i][j] + matrix_B[i][j])
        matrix_sum.append(row)
    return matrix_sum



def matrix_subtraction(matrix_A,matrix_B):
    """
    Returns the difference between two equal-sized matrices. If the input parameter matrices are not of equal
    size nothing is returned.
    """
    m_1,n_1 = get_size(matrix_A)
    m_2,n_2 = get_size(matrix_B)

    if m_1 != m_2 or n_1 != n_2:
        return

    matrix_sum = []
    for i in range(m_1):
        row = []
        for j in range(n_1):
            row.append(matrix_A[i][j] - matrix_B[i][j])
        matrix_sum.append(row)
    return matrix_sum



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

    B_transposed  = transpose_matrix(matrix_B)

    out = []
    for row in matrix_A:
        tmp = []
        for i in range(cols_B):
            val = 0
            col_to_multiply = B_transposed[i]
            for j in range(len(col_to_multiply)):
                val += row[j] * col_to_multiply[j]
            tmp.append(val)
        out.append(tmp)   
    return out



def create_identity_matrix(order: int):
    """
    Returns the identity matrix of a given order as a list of lists.
    """
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



def reduced_row_echelon(augmented_matrix):
    """

    """
    m,n = [len(augmented_matrix),len(augmented_matrix[0])]
    n_variables = n - 1
    evaluated_rows = []

    for i in range(n_variables):
        maxrow = 0
        maxval = 0

        for j in range(m):
            if (abs(augmented_matrix[j][i]) > abs(maxval)) and j not in evaluated_rows:
                maxrow = j
                maxval = augmented_matrix[j][i]
        evaluated_rows.append(maxrow)

        if maxval == 0:
            continue

        other_rows = [row for row in range(m) if row != maxrow]
        reciprocal = 1 / augmented_matrix[maxrow][i]
        new_row = [coefficient * reciprocal for coefficient in augmented_matrix[maxrow]]

        augmented_matrix[maxrow] = new_row

        for row_num in other_rows:
            multiplier = augmented_matrix[row_num][i]
            new_other_row = [augmented_matrix[row_num][k] - (multiplier * new_row[k]) for k in range(n)]
            augmented_matrix[row_num] = new_other_row

    augmented_matrix = tidy_up(augmented_matrix)

    if n_variables > m:
        n_exchanges = m
    else:
        n_exchanges = n_variables

    for variable in range(n_exchanges):
        for i in range(m):
            if (augmented_matrix[i][variable] != 0) and (i != variable):
                tmp = augmented_matrix[variable]
                augmented_matrix[variable] = augmented_matrix[i]
                augmented_matrix[i] = tmp
                break
   
    return augmented_matrix



def invert_matrix(matrix):
    """
    Takes a matrix as parameter, returns nothing if parameter matrix is nonsquare and thus non-invertible.
    The function then proceeds to check if the equation system has exactly one solution, and if not it will return 
    nothing as the matrix is non-invertible. 
    
    Finally it will adjoin the input parameter matrix with its corresponding identity matrix and reduce it with
    Gauss-Jordan elimination in order to return the inverted matrix.
    """
    m,n = get_size(matrix)
    if m != n: # nonsquare matrices do not have inverses
        return
    identity_matrix = create_identity_matrix(m)
    
    # only matrices with exactly one solution are invertible, so lets use our gauss-jordan script
    reduced_matrix = [row for row in matrix]
    reduced_matrix = reduced_row_echelon(reduced_matrix)
    
    for row in reduced_matrix: # check for zero-row
        invertible = False
        for value in row:
            if value != 0:
                invertible = True
        if not invertible:
            print('not invertible')
            return
    
    adjoined_matrix = []
    for i in range(m):
        tmp = [val for val in matrix[i]]
        tmp.extend(identity_matrix[i])
        adjoined_matrix.append(tmp)

    reduced_adjoined_matrix = reduced_row_echelon(adjoined_matrix)
    inverted_matrix = []
    for i in range(m):
        inverted_matrix.append([reduced_adjoined_matrix[i][j] for j in range(n,(2*n))])
    return inverted_matrix


"""
APPLICATIONS
"""

def polynomial_curve_fitting(matrix):
    """
    This function takes a list of nested lists as its input parameter
    where each nested list should have two numeric value entries representing
    an x,y coordinate - read_matrix() can be used to this end. 

    The function sorts the input coordinates in ascending order, chooses an n-degree polynomial
    where n is equal to one less than the number of coordinates and creates as system of linear
    equations on the basis of the coordinates and the polynomial as an augmented matrix of the system.

    Solving the equations yields the polynomial function which is returned as a list of coefficients of the form
    p(x) = a_0 + a_1x + a_2x^2 + ... + a_nx^n
    """
    matrix.sort()
    n_coordinates = len(matrix)

    augmented_matrix = []
    for i in range(n_coordinates):
        row = [1]
        x,y = matrix[i]
        for j in range(1,n_coordinates):
            row.append(x ** j)
        row.append(y)
        augmented_matrix.append(row)

    augmented_matrix = reduced_row_echelon(augmented_matrix)
    return [row[-1] for row in augmented_matrix]



def least_squares_regression(matrix):
    """
    This function takes a list of nested lists as its input parameter
    where each nested list should have two numeric value entries representing
    an x,y coordinate - read_matrix() can be used to this end. 
    
    The function then sorts the coordinates in ascending order and performs the regression,
    returning a list containing the slope a, intercept b and sum of squared errors from the regression line y = b + ax + e
    """
    matrix.sort()

    X = [[1,coordinate[0]] for coordinate in matrix]
    X_transposed = transpose_matrix(X)

    # Calculate X^T * X
    XTX = dot_product(X_transposed,X)

    # Calculate X^T * Y
    Y = [[coordinate[1]] for coordinate in matrix]
    XTY = dot_product(X_transposed,Y)

    # Invert XTX, define slope coefficient a and intercept b
    XTX_inverted = invert_matrix(XTX)
    results = tidy_up(dot_product(XTX_inverted,XTY))
    a = results[1][0]
    b = results[0][0]

    # Calculate sum of squared errors
    error_sum = 0
    for coordinate in matrix:
        error_sum += ((b + coordinate[0] * a) - coordinate[1]) ** 2
    return [b,a,error_sum]




def main():
    A = read_matrix('test_data/test_matrix_addition/2.1_ex2-1.txt')
    for line in A:
        print(line)
    print()
    B = read_matrix('test_data/test_matrix_addition/2.1_ex2-2.txt')
    
    for line in matrix_addition(A,B):
        print(line)


if __name__ == '__main__':
    main()
