# Header define
yahoo_finance_headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'no-cache',
        'Cookie': 'A1=d=AQABBJkNsWgCEEi_58VJbGB_l7OJZ7XWrc4FEgEBAQFfsmi6aK8AAAAA_eMCAA&S=AQAAAl61G0LkQqr0GzwTZZlMD30; A3=d=AQABBJkNsWgCEEi_58VJbGB_l7OJZ7XWrc4FEgEBAQFfsmi6aK8AAAAA_eMCAA&S=AQAAAl61G0LkQqr0GzwTZZlMD30; A1S=d=AQABBJkNsWgCEEi_58VJbGB_l7OJZ7XWrc4FEgEBAQFfsmi6aK8AAAAA_eMCAA&S=AQAAAl61G0LkQqr0GzwTZZlMD30; PRF=t%3DTSLA; gpp=DBAA; gpp_sid=-1; fes-ds-session=pv%3D7; cmp=t=1756438244&j=0&u=1---',
        'Priority': 'u=0, i',
        'Sec-Ch-Ua': '"Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36'
    }

# Define import function
def us_stock_price_yahoo_finance(ticker = 'TSLA', period1 = '1262204200', period2='1756433964', ingest_header = yahoo_finance_headers):
    # Send request
    response = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?&includeAdjustedClose=true&interval=1d&period1={period1}&period2={period2}",
                           headers = ingest_header)
    data = response.json()
    # navigate to the data
    result = data['chart']['result'][0]
    timestamps = result['timestamp']
    quotes = result['indicators']['quote'][0]
    adjclose = result['indicators']['adjclose'][0]['adjclose']
    
    # build dataframe
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps, unit='s'),
        'open': quotes['open'],
        'high': quotes['high'],
        'low': quotes['low'],
        'close': quotes['close'],
        'adjclose': adjclose,
        'volume': quotes['volume']
    })

    # add custom date fields
    df['datetime'] = df['timestamp'].dt.strftime('%Y%m%d')
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['ticker'] = ticker
    df['table_name'] = 'us_stock_price_yahoo_finance'
    
    return(df)