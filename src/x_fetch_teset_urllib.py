import urllib.robotparser

rp = urllib.robotparser.RobotFileParser()
rp.set_url("https://www.hackingwithswift.com/robots.txt")
rp.read()
print(
    rp.can_fetch(
        "SvenBrowser/1.0 (anton@mimecam.com)",
        "https://www.hackingwithswift.com/articles/278/whats-new-in-swiftui-for-ios-26",
    )
)
exit(0)
