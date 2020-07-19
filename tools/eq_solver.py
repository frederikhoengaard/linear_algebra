import utilities

augmented_matrix = utilities.read_matrix('d.txt')
m,n = [len(augmented_matrix),len(augmented_matrix[0])]

for line in augmented_matrix:
    print(line)

n_variables = n - 1

evaluated_rows = []

for i in range(n_variables):
    print('\n   ===============================')                       # can be deleted later
    print('   === Evaluating variable x_'+str(i+1),'===')               # can be deleted later
    print('   ===============================\n')                       # can be deleted later
    
    print('      - Augmented matrix currently looks like:\n')           # can be deleted later
    for line in augmented_matrix:                                       # can be deleted later
        print('       ',line)                                           # can be deleted later
    print()
    maxrow = 0
    maxval = 0

   # print('\n       Evaluated rows',evaluated_rows)
    for j in range(m):
        if (abs(augmented_matrix[j][i]) > abs(maxval)) and j not in evaluated_rows:
            maxrow = j
            maxval = augmented_matrix[j][i]
    evaluated_rows.append(maxrow)

    if maxval == 0:
        continue

    current_row = maxrow
    other_rows = [row for row in range(m) if row != current_row]

    #first reduce current row so that relevant variable gets coefficient 1
    print('      - The row with the numerically largest coefficient for x_'+str(i+1),'is row',maxrow)                       # can be deleted later
    reciprocal = 1 / augmented_matrix[current_row][i]
    print('      - The coefficient for x_'+str(i+1),'in row',maxrow,'is',maxval,'to which the reciprocal is',reciprocal)    # can be deleted later
    print('      - We can then multiply row',maxrow,'by',reciprocal,'in order to set the x_'+str(i+1)+'-value equal to 1, and the augmented matrix becomes:\n')  # can be deleted later
    newrow = [value * reciprocal for value in augmented_matrix[current_row]]
    augmented_matrix[current_row] = newrow

    for line in augmented_matrix:                                       # can be deleted later
        print('       ',line)                                           # can be deleted later

    #then subtract newrow from other rows in order to eliminate other coefficients
    for row_num in other_rows:       
        multiplier = augmented_matrix[row_num][i]
      #  print(row_num,augmented_matrix[row_num],multiplier)
        new_other_row = [augmented_matrix[row_num][k] - (multiplier * newrow[k]) for k in range(n) ]
        augmented_matrix[row_num] = new_other_row
        print('\n      - We subtract row',maxrow,'from row',row_num,multiplier,'times to achieve\n')
        for line in augmented_matrix:                                       # can be deleted later
            print('       ',line)                                           # can be deleted later
    print()



    