# The History of Search Engines

Early search engines emerged in the 1990s as the World Wide Web expanded beyond academic networks. Tools like Archie, Veronica, and Jughead indexed FTP archives and Gopher menus, giving users their first taste of automated information retrieval. These systems were primitive by modern standards, relying on exact string matching with no notion of relevance ranking.

## The Rise of Web Crawlers

The mid-1990s saw a wave of web crawlers that changed the landscape. AltaVista launched in 1995 and quickly became the most popular search engine, indexing millions of pages and introducing features like natural language queries. Lycos, Excite, and Infoseek competed for users by adding directory categories and editorial curation alongside algorithmic results.

WebCrawler was the first engine to index full page text rather than just titles and headers. This seemingly simple change dramatically improved recall, since users could now find pages that mentioned their query terms anywhere in the body. The downside was noise: irrelevant pages that happened to contain the right words cluttered results.

## PageRank and the Google Era

Larry Page and Sergey Brin introduced PageRank in 1998, fundamentally shifting search from keyword matching to link-based authority scoring. The insight was elegant: a page linked to by many other pages is probably more important, and links from important pages count more than links from obscure ones. This recursive definition created a global ranking signal that was remarkably resistant to simple manipulation.

Google combined PageRank with traditional term-frequency signals and a clean interface. By 2000, it had overtaken AltaVista, Yahoo, and every other competitor. The search engine wars were effectively over for a decade.

## Modern Semantic Search

Today, search has moved beyond lexical matching entirely. Dense retrieval models encode queries and documents into high-dimensional vector spaces, where semantic similarity replaces exact keyword overlap. A query for "how to fix a leaky faucet" can match a document titled "plumbing repair guide" even if the exact words differ. Hybrid approaches combine vector search with traditional BM25 scoring, using reciprocal rank fusion to merge the two result lists.
