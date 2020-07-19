import utilities 

augmented_matrix = utilities.read_matrix('a.txt')
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

for variable in range(n_variables):
   for i in range(m):
      if (augmented_matrix[i][variable] != 0) and (i != variable):
         tmp = augmented_matrix[variable]
         augmented_matrix[variable] = augmented_matrix[i]
         augmented_matrix[i] = tmp
         break

for line in augmented_matrix:
   print(line)