import json

def json2text(path, input_id, lang_tag):
    '''
    function: extract text of corresponding id and lang
    parameters:
             path: path of json file
             input_id: sign video id
             lang_tag: the language of target text
    '''
    with open(path, 'r', encoding='utf-8') as f:
        # type of data is list, every sample in data is dict
        data = json.load(f)
        # search dictionary whose id = input_id
        dictionary = dictList2dict(data, input_id)
        
        print("input_id lang_tag %s" % input_id, "lang_tag %s"% lang_tag)
        # type of signTextlist is list, every sample in data is dict
        signTextlist = dictionary['sign_list']
        signDict = dictList2dict(signTextlist, lang_tag)
        target_text = signDict['text']
        
    return target_text



def dictList2dict(l, input_value):
    '''
    function: find wanted dict in dict list like {dict_1,dict_2,...,dict_n}
    '''
    for dictionary in l:
        v_list = dictionary.values()
        if input_value in v_list:
            return dictionary




if __name__=='__main__':
    path = '/home/gzs/baseline/sp-10/dataset/train.json'
    text = json2text(path, 410, 'sv')
    print(text)
