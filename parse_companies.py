import re
import yaml
from pathlib import Path

# Company names from the README
companies_text = """23andMe,6sense,A9,Academia,Achievers,Acorns,Acquia,Addepar,Affirm,Ahrefs,Airbnb,Airtable,Algebraix,Algolia,AltSchool,Amino,Amobee,Ancestry,ANDigital,Angellist,AppFolio,AppNexus,Apptio,Aptible,Asana,Aspiration Software,Atomic,August,AutomatedInsights,Automattic,BambooHR,Benchling,Bending Spoons,Betterment,Bill.com,Bitly,Bitnami,BitTiger,BitTorrent,Blend Labs,Bloomz,Blue Origin,Bonuz,Booking.com,Box,Brigade,BuildASign,BuildZoom,Bungie,Busbud,BuzzFeed,Canary,Capsule,Captain401,Carbon,Carousell,Casetext,Casper,Catchpoint,Centro,change.org,Chaordic,Checkr,Cheesecake Labs,Chewy,Chexology,CircleUp,Citrix,Classy,ClearSlide,Clever,Clippings,Cloudera,Clue,Clustrix,Codecademy,Codefights,Codeship,Coinbase,Collective Health,Color,Compass,CoreOS,CornerStone OnDemand,Couchbase,Couchsurfing,Counsyl,Coursera,CreditKarma,CrowdRise,CrowdStar,Cruise,CrunchBase,Cumulus Networks,Curse,Curalate,Datadog,DigitalOcean,Disqus,DJI,Docker,DocuSign,DollarShaveClub,DoorDash,Dot & Bo,DOT Digital Group,DraftKings,DroneDeploy,Dropbox,Duolingo,EagleView,Edmodo,Edmunds,EdX,eHarmony,Elasticsearch,Element.ai,Elemental Technologies,EnergySavvy,Entelo,Epic,Etsy,Eventbrite,Evernote,Facebook,Farfetch,Faraday Future,Fastly,Fictiv,Fitbit,FiveStars,Flatiron,Flexport,Flipboard,Flipkart,Flipp,Forex Factory,Foursquare,frame.ai,Funding Circle,General Assembly,GetYourGuide,Gigster,GitHub,GitLab,GoDaddy,Gree,Greenhouse,Groupon,GrubHub,Guidewire,GumGum,Gusto,HackerRank,Hailo,Haiku Deck,Hart Inc.,Helix,Helpling,Hipmunk,HomeAway,HubSpot,Hulu,iCIMS,Idealab,IFTTT,IgnitionOne,Illumio,Imgur,Indeed,Indiegogo,Infinera,InfluxDB,InMobi,InstaCart,Instructure,Integral Ad Science,Intellisis,Intentional,Interactive Intelligence,Intercom,Intrepid Pursuits,Invoice2go,IXL Learning,Jawbone,Jet,JetBrains,Jiminny,Jive,Jobvite,JustWatch,Kayak,KeepSafe,Khan Academy,Kickstarter,Kik,Kinnek,Klarna,Knowles,Laserfiche,LendUp,Lever,Lifesize,Liftoff,LinkedIn,Linode,Lithium,Livescribe,Logitech,Looker,Lucid Software,Luxe,Lyft,Machine Zone,Magic,MailChimp,Marketo,Mark 43,Mattermark,Matterport,Medallia,Medium,Meetup,Mesosphere,Meteor,Metromile,Mind Candy,MindTouch,MixBit,Mixmax,Mixpanel,MLab,MongoDB,Moz,Mozilla,MuleSoft,Munchery,Narrative Science,Nervana Systems,Nest,New Relic,Nexmo,Nextdoor,Niantic,Nulogy,Nutanix,Oblong,Okta,Onefootball,OneLogin,Open Whisper Systems,Opendoor,OpenTable,OpenX,Operator,Optimizely,Oscar Health,Palantir,Palo Alto Networks,Pandora,Paypal,Paytm Labs,Pinterest,Pivotal,PlushCare,Postmates,Prezi,Priceline,Prism,Pristine,Procore Technologies,Prominic,Pure Storage,Qlik,Qualtrics,Quantcast,Qubole,Quip,Quizlet,Quora,Rackspace,Radius,ReadMe,Redbooth,Redbull,Reddit,Redfin,RedMart,Redis-Labs,Red Nova Labs,Remitly,ResearchGate,Resultados Digitais,RetailNext,Riot Games,RiskIQ,Roadtrippers,Robinhood,ROBLOX,Roku,Rover.com,rubrik,SADA Systems,ScoreBig,Scribd,Seamless,SeatGeek,Segment,SendGrid,Sensus,Shazam,SheerID,Shopify,ShoreTel,Shutterfly,Sift Science,Signifyd,SimplyCredit,SKHMS,Skillshare,Slack,Slice,Smarking,Smartsheet,SmartThings,Smule,Snapchat,SocialBase,Socotra,Source Intelligence,SoundCloud,SpaceX,Spotify,Sprout Social,Square,Squarespace,SSi Micro,StackPath,SteelHouse,Stitch Fix,Stripe,Study.com,Sunlink,SurveyMonkey,Symphony,Tableau,Taco Bell Corporate,Takt,Tango,TED,Teespring,Telenav,TextNow,The Artist Union,The Climate Corporation,The Internet Archive,The League,ThoughtSpot,ThoughtWorks,Thumbtack,Tile,Tillster,Tinder,Tint,TiVo,Top Hat,Trail,Travis CI,TripAdvisor,Trustpilot,Trustwave,Tumblr,Twilio,Twitch,Twitter,Two Roads,Two Sigma,Udacity,Unity,Upwork,Urban Massage,Valve,Veeva,Venmo,Vertafore,VEVO,VHX,Viget,Vimeo,Vox Media,Vultr,Wattpad,WayUp,Wealthfront,Wealthsimple,Weebly,Whitepages,Wikimedia Foundation,Wish,Wolt,Work Market,Workiva,Xero,Yelp,Yext,YouNow,Zalando,Zappos,Zendesk,Zenefits,ZenMate,Zillow,ZocDoc,Zoosk,Zscaler,Zuora,Zynga"""

# Parse and convert to slugs
companies = []
for name in companies_text.split(','):
    name = name.strip()
    # Convert to slug
    slug = name.lower().replace(' ', '-').replace('.', '')
    slug = re.sub(r'[^\w-]', '', slug)
    slug = re.sub(r'-+', '-', slug)
    slug = slug.strip('-')
    companies.append(slug)

print(f"Extracted {len(companies)} companies")
print(f"First 10: {companies[:10]}")
print(f"Last 10: {companies[-10:]}")

# Load current config
config_path = Path('profiles/default/config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Get current Greenhouse companies
current_gh = config.get('sources', {}).get('greenhouse', {}).get('companies', [])
print(f"\nCurrent Greenhouse companies: {len(current_gh)}")
print(f"  {current_gh[:5]}")

# Combine: existing first, then new ones  
existing_set = set(current_gh)
new_companies = [c for c in companies if c not in existing_set]

combined = current_gh + new_companies
print(f"\nAfter adding all new companies: {len(combined)} total")
print(f"  Added {len(new_companies)} new companies")

# Update config — ensure all nested dicts exist
if 'sources' not in config:
    config['sources'] = {}
if 'greenhouse' not in config['sources']:
    config['sources']['greenhouse'] = {}
if 'enabled' not in config['sources']['greenhouse']:
    config['sources']['greenhouse']['enabled'] = True

config['sources']['greenhouse']['companies'] = combined

# Save
with open(config_path, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"\n✓ Updated config.yaml with {len(combined)} Greenhouse companies")
