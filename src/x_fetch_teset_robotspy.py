import robots

parser = robots.RobotsParser.from_uri(
    "https://www.hackingwithswift.com/robots.txt"
)
useragent = "Nutch"
path = "/articles/"
result = parser.can_fetch(useragent, path)
print(f"Can {useragent} fetch {path}? {result}")
exit()
