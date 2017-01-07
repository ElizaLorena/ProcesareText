def getNextTag(posTag, index, whatIs):
        try:
            i = index
            while(i < len(posTag)):
                i += 1
                tag = posTag[i]
                if tag[1] in whatIs:
                    i += 1
                    while(i<len(posTag)):
                        if posTag[i] not in whatIs:
                            break
                        else:
                            tag = posTag[i]
                            i+=1
                    return (tag, i - 1)
            return (None, None)
        except Exception:
            return (None, None)