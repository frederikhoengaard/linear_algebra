# utilities.py

def read_matrix(filename):
   infile = open(filename,'r')
   return [list(map(float,row.split())) for row in infile.readlines()]



def tidy_up(matrix):
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
   for i in range(len(matrix)):
      for j in range(len(matrix[i])):
         matrix[i][j] *= scalar
   return matrix



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
