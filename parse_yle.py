import os
import json
from datetime import datetime, date

languages = ['fi', 'sv']

def process_articles(path):
    articles = {lang: [] for lang in languages}
    for lang in languages:
        print("Processing articles in: ", lang)
        path_lang = path+"/"+lang
        years = os.listdir(path_lang)
        years.sort()
        for y in years:
            path_year = path_lang+"/"+y
            months = os.listdir(path_year)
            months.sort()
            for m in months:
                path_month = path_year+"/"+m
                files = os.listdir(path_month)
                files.sort(reverse=True)
                for f in files:
                    path_file = path_month+"/"+f
                    json_file = open(path_file,'r')
                    json_dict = json.load(json_file)
                    data = json_dict['data']
                    print("Processing file:", path_file)
                    for art in data:
                        date_pub = art['datePublished']
                        date_pub = date_pub.split("-")
                        year_pub = date_pub[0]
                        month_pub = date_pub[1]
                        day_pub = date_pub[-1].split("T")[0]
                        date_formatted = date(year=int(year_pub), month=int(month_pub), day=int(day_pub))
                        headline = art['headline']['full']
                        art_content = art['content']
                        content = ""
                        for con in art_content:
                            if 'type' in con.keys():
                                if con['type'] == 'text':
                                    content += con['text'] + " "
                        subject_list = []
                        if "subjects" in art.keys():
                            subjects = art['subjects']
                            for sub in subjects:
                                #subject_list.append(sub['title']['fi'])
                                subject_list.append(sub['title']['sv'])
                        a = {"date": date_formatted, "headline": headline, "content": content}
                        if len(subject_list) > 0:
                            a['subjects'] = subject_list
                        articles[lang].append(a)
    return articles

def align_articles(articles):
    #align articles using date and named entities
    aa = {lang: [] for lang in languages}
    unmatched = {lang: [] for lang in languages}
    aa_count = 0
    un_count = 0
    for art_fi in articles['fi']:
        date_fi = art_fi['date']
        #print(date_fi)
        if date_fi.year:
            if 'subjects' in art_fi.keys():
                subjects_fi = [s for s in art_fi['subjects'] if s is not None]
                for art_sv in articles['sv']:
                    day_delta = (art_sv['date'] - date_fi).days
                    if abs(day_delta) <= 2: #check Swedish articles published 2 days before/after the Finnish article
                        #extract relevant NE from the Swedish article
                        text_sv = art_sv['content']
                        subjects_sv = [s for s in subjects_fi if s in text_sv]
                        #check if the articles share 3 or more NEs
                        inter = list(set(subjects_fi).intersection(set(subjects_sv)))
                        if len(subjects_sv) >= 5:
                            aa['fi'].append(art_fi)
                            aa['sv'].append(art_sv)
                            aa_count += 1
                            print(date_fi)
                            print("Aligned articles:", aa_count)
                            articles['sv'].remove(art_sv)
                            break
                        # store unmatched articles for validation/testing
                        else:
                            unmatched['fi'].append(art_fi)
                            unmatched['sv'].append(art_sv)
                            un_count += 1
                            #print("Unmatched articles: ", un_count)
                    elif day_delta >= 30:
                        break
    print("Total aligned articles: ", aa_count)
    print("Total unmatched articles: ", un_count)
    return aa, unmatched


def write_articles_to_file(path):
    fp = open(path, "r")
    data = json.load(fp)
    languages = list(data.keys())
    text_data = {lang: {} for lang in languages}
    art_count = len(data['fi'])
    for i in range(art_count):
        print("Art count: ", i)
        date = data['fi'][i]['date']
        header = "||ArticleNo:" + str(i + 1)+"||"
        text_data_fi = data['fi'][i]['content']
        text_data_sv = data['sv'][i]['content']
        if date in text_data['fi'].keys():
            text_data['fi'][date] += "\n" + header + text_data_fi
            text_data['sv'][date] += "\n" + header + text_data_sv
        else:
            text_data['fi'][date] = header + text_data_fi
            text_data['sv'][date] = header + text_data_sv
    #write articles to text files
    dates = text_data['fi'].keys()
    parent_dir = "data/yle/raw_text/"
    for dat in dates:
        print("Date: ", dat)
        for lang in languages:
            fname = parent_dir+dat+"_"+lang+".txt"
            fp = open(fname, 'w')
            fp.write(text_data[lang][dat])
            fp.close()
            print("Saved file as: ", fname)
    print("Done writing all articles as raw text!")


path = "data/yle"
articles = process_articles(path)
aa, unmatched = align_articles(articles)

len_aa = len(aa['fi'])
for i in range(len_aa):
    d = aa['fi'][i]['date']
    aa['fi'][i]['date'] = d.strftime('%Y-%m-%d')
    d = aa['sv'][i]['date']
    aa['sv'][i]['date'] = d.strftime('%Y-%m-%d')

with open('yle_aligned.json', 'w') as f:
    json.dump(aa, f)

len_unmatched = len(unmatched['fi'])
for i in range(len_unmatched):
    d = aa['fi'][i]['date']
    aa['fi'][i]['date'] = d.strftime('%Y-%m-%d')
    d = aa['sv'][i]['date']
    aa['sv'][i]['date'] = d.strftime('%Y-%m-%d')åå

for i in range(20):
    print("FI: ", aa['fi'][i]['headline'])
    print("SV: ", aa['sv'][i]['headline'])
    print("\n")