import utilities

A = utilities.read_matrix('A3.txt')
B = utilities.read_matrix('B3.txt')

def dot_product(matrix_A,matrix_B):
    # B must have as many rows as A has columns
    cols_A = len(matrix_A[0])
    rows_B = len(matrix_B)
    cols_B = len(matrix_B[0])

    if cols_A != rows_B:
        return

    B_columns = []
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

for line in dot_product(A,B):
    print(line)