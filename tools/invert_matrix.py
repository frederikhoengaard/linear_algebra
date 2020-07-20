import utilities
import equation_system_solver

def invert_matrix(matrix):
   m = len(matrix)
   n = len(matrix[0])
   if m != n: # nonsquare matrices do not have inverses
      return
   identity_matrix = utilities.create_identity_matrix(m)
   
   # only matrices with exactly one solution are invertible, so lets use our gauss-jordan script
   reduced_matrix = [row for row in matrix]
   reduced_matrix = equation_system_solver.reduced_row_echelon(reduced_matrix)
   
   for row in reduced_matrix: # check for zero-row
      invertable = False
      for value in row:
         if value != 0:
            invertable = True
      if not invertable:
         print('not invertable')
         return
   
   adjoined_matrix = []
   for i in range(m):
      tmp = [val for val in matrix[i]]
      tmp.extend(identity_matrix[i])
      adjoined_matrix.append(tmp)

   reduced_adjoined_matrix = equation_system_solver.reduced_row_echelon(adjoined_matrix)
   inverted_matrix = []
   for i in range(m):
      inverted_matrix.append([reduced_adjoined_matrix[i][j] for j in range(n,(2*n))])
   return inverted_matrix
      

   


A = utilities.read_matrix('i6.txt')

if invert_matrix(A):
   for line in invert_matrix(A):
      print(line)