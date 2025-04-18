path = 'grapho_terrain/telecommunications/coverage.py'
with open(path, 'r') as f: lines = f.readlines()
for i in range(len(lines)):
    if lines[i].strip() == 'try:':
        if i+1 < len(lines) and not lines[i+1].startswith('        '):
            indentation = lines[i].split('try:')[0]
            lines[i+1] = indentation + '    ' + lines[i+1].lstrip()
            break
with open(path, 'w') as f: f.writelines(lines)
print('File updated successfully!')
