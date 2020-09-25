

bins = ''
for i in range(-10790, 8440, 150):
    temp = '({}, {}), '.format(i, i+150)
    bins = bins+temp

print(bins)
