# Requires transformers>=4.48.0
from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "Alibaba-NLP/gte-reranker-modernbert-base",
    automodel_args={"torch_dtype": "auto"},
)

q = 'view in SwiftUI'

docs = [
    'do bond funds pay dividends',
    "A bond fund or debt fund is a fund that invests in bonds, or other debt securities. Bond funds can be contrasted with stock funds and money funds. Bond funds typically pay periodic dividends that include interest payments on the fund's underlying securities plus periodic realized capital appreciation. Bond funds typically pay higher dividends than CDs and money market accounts. Most bond funds pay out dividends more frequently than individual bonds.",
    'You would have $71,200 paying out $1,687 in annual dividends. That is about $4.62 for working up in the morning. Interestingly enough, that 2.37% yield is at a low point because The Wellington Fund is a meaning that it holds a combination of stocks and bonds.',

    'Matrix is a movie from Hollywood starring Neo and Trinity',
    'Matrix part 2 was filmed in Europe as well as Los Angeles',
    'Intern is a Hollywood movie about a retired man in NY city becoming an intern in an Internet company',

    '```swift import Foundation final class UsernameGenerator { static func next() -> String {let pair = nextPair(); return "\\(pair.0)_\\(pair.1)"} private static func nextPair() -> (String, String) {(first[Int.random(in: 0..<first.count)].lowercased(),second[Int.random(in: 0..<second.count)].lowercased())}}```',
    'What class can generate a unique username in swift lang?',
    'struct DarkAlertView: View { var body: some View { Text(\'hello, SwiftUI!\') } }',
]


results = model.rank(
    q, docs, 
    # return_documents=True, 
    top_k=5
)
print(results)


