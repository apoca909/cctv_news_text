from playwright.sync_api import sync_playwright

def getDates(start, end):
    import datetime
    dates = []
    datestart = datetime.datetime.strptime(start, '%Y%m%d')
    while datestart <= datetime.datetime.strptime(end, '%Y%m%d'):
        dates.append(datestart.strftime('%Y%m%d'))
        datestart += datetime.timedelta(days=1)
        
    return dates

def fetch_cctv_news(date='20080101'):
    title_conts = []
    with sync_playwright() as playwright:
        browser = playwright.firefox.launch(headless=True, )
        context = browser.new_context()
        page = context.new_page()
        
        page.goto(f'https://cn.govopendata.com/xinwenlianbo/{date}/')
        if page.title() == '':
            return title_conts
        
        node_title_cont = page.query_selector_all("div[class='col-md-9 col-sm-12 heti']")[0]
        node_titles = node_title_cont.query_selector_all("h2[class='h4']")
        node_contents = node_title_cont.query_selector_all("p")

        for i, t in enumerate(node_titles):
            title = t.inner_text()
            cont  = node_contents[i].inner_text()
            title_conts.append({'title':title, 'content':cont, })
    return title_conts

if __name__ == '__main__':
    for d in getDates('20080101', '20080109'):
        rs = fetch_cctv_news(d)
        print(d, rs[0])
