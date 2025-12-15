import os

for filename in ['src/ui/app.py', 'src/core/cli.py', 'src/reports/pdf_report.py', 'src/reports/excel_export.py']:
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        if content.startswith('\ufeff'):
            content = content[1:]
        content = content.replace('\\"', '"')
        with open(filename, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)
        print(f'Fixed {filename}')
    else:
        print(f'Skipped {filename} (not found)')
