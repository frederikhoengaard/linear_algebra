import utilities 

def reduced_row_echelon(augmented_matrix):
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

   augmented_matrix = utilities.tidy_up(augmented_matrix)

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

def main():
   A = utilities.read_matrix('non_invert.txt')
   for line in reduced_row_echelon(A):
      print(line)


if __name__ == '__main__':
   main()
