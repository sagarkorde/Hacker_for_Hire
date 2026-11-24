#!/usr/bin/env sh
# Downloads the public Agora dataset and the daily BTC/USD series used in the paper from their GitHub
# mirrors and checks both files against the SHA-256 values of the copies we analysed.
set -e
cd "$(dirname "$0")"
curl -L -o Agora.csv https://raw.githubusercontent.com/russellkim98/DarkNetAnalysis/master/Agora.csv
curl -L -o btc_usd_daily.csv https://raw.githubusercontent.com/davide-perez/datavisexp-agora/main/data/BTCHist01012014.csv
printf "%s  Agora.csv\n%s  btc_usd_daily.csv\n" 743c05fa682d3c940e80ed03fe52c6bc3035236c2f3fddc09c59baa797a745a2 a3176f4f67c8424b14f35efedf18052d676bc01b1b974c7ab9b09e929077c72a | sha256sum -c -
