import yaml

with open('profiles/default/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

gh = config.get('sources', {}).get('greenhouse', {})
print(f'greenhouse type: {type(gh)}')
print(f'greenhouse keys: {list(gh.keys())}' if isinstance(gh, dict) else 'not a dict')
print()
companies = gh.get('companies', [])
print(f'companies type: {type(companies)}')
print(f'companies length: {len(companies)}')
if companies:
    print(f'First item: {repr(companies[0])} (type: {type(companies[0]).__name__})')
    print(f'Is first item a string? {isinstance(companies[0], str)}')
