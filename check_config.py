import yaml

with open('profiles/default/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

gh_companies = config.get('sources', {}).get('greenhouse', {}).get('companies', [])
print(f'Greenhouse companies count: {len(gh_companies)}')
print(f'Type: {type(gh_companies)}')
print(f'First 5: {gh_companies[:5]}')
if gh_companies:
    print(f'Type of first item: {type(gh_companies[0])}')
    print(f'First item value: {repr(gh_companies[0])}')
