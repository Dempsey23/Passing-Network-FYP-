from TOT_VS_UNITED import A_TOT as TOT_UNI_ADJ
from TOT_VS_STOKE import A_TOT as TOT_STO_ADJ
from TOT_VS_NEWCASTLE import A_TOT as TOT_NEW_ADJ
import numpy as np
from TOT_VS_LEICESTER import A_TOT as TOT_LEI_ADJ

sourceFile=open("Similarity_Between_Matches_brute_forceTOT.txt",'w')
min_value = min(np.min(TOT_UNI_ADJ), np.min(TOT_STO_ADJ),np.min(TOT_NEW_ADJ))
max_value = max(np.max(TOT_UNI_ADJ), np.max(TOT_STO_ADJ),np.max(TOT_NEW_ADJ))

# Normalize both matrices to the range [0, 1]
normalized_TOT_UNI = (TOT_UNI_ADJ - min_value) / (max_value - min_value)
normalized_TOT_NEW = (TOT_NEW_ADJ - min_value) / (max_value - min_value)
normalized_TOT_STO = (TOT_STO_ADJ - min_value) / (max_value - min_value)
def sum_of_absolute_differences_by_row(matrixA, matrixB):
    # Check if matrices have the same dimensions
    if len(matrixA) != len(matrixB) or len(matrixA[0]) != len(matrixB[0]):
        raise ValueError("Matrices must have the same dimensions")

    total_sum = 0
    # Iterate over rows
    for i in range(len(matrixA)):
        row_sum = 0  # Initialize sum for the current row
        # Iterate over columns
        for j in range(len(matrixA[0])):
            # Calculate the absolute difference for the current cell
            row_sum += abs(matrixA[i][j] - matrixB[i][j])

        # Add row sum to total sum
        total_sum += row_sum
        print(f"Sum after row {i + 1}: {total_sum}")
        print(f"    Sum of this row {i + 1}: {row_sum}")
    return total_sum

print('Similarity TOT passing VS UNITED & VS NEWCASTLE (WIN/LOSS)\n---------------\n',sum_of_absolute_differences_by_row(normalized_TOT_UNI,normalized_TOT_NEW),file=sourceFile)
print(sum_of_absolute_differences_by_row(normalized_TOT_UNI,normalized_TOT_NEW))

print('Similarity TOT passing VS STOKE & VS NEWCASTLE (DRAW/LOSS) \n---------------\n',sum_of_absolute_differences_by_row(normalized_TOT_STO,normalized_TOT_NEW),file=sourceFile)
print(sum_of_absolute_differences_by_row(normalized_TOT_STO,normalized_TOT_NEW))

print('Similarity TOT passing VS STOKE & VS UNITED (DRAW/WIN) \n---------------\n',sum_of_absolute_differences_by_row(normalized_TOT_STO,normalized_TOT_UNI),file=sourceFile)
print(sum_of_absolute_differences_by_row(normalized_TOT_STO,normalized_TOT_UNI))
