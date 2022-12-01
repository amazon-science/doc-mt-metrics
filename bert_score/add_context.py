import argparse
from typing import List


def add_context(org_txt: List[str], context: List[str], docs: List[str], sep_token: str = "</s>",
                ws: int = 2) -> List[str]:
    """Function that adds the previous sentences as context to the current sentence, respecting document boundaries
    :param org_txt: the original text
    :param context: the text from which the context will be taken (same as org_txt for source/reference)
    :param docs: the document where each segment belongs to
    :param sep_token: the separator token of the tokenizer for the specific model
    :param ws: the window size, maximum of the previous sentences to be considered as context
    :return: the original text augmented with context
    """
    i, k = 0, 0
    augm_txt = []
    doc_id = docs[0]
    while i < len(org_txt):
        if docs[i] == doc_id:
            context_window = context[i - min(k, ws):i]
            augm_txt.append(" {} ".format(sep_token).join(context_window + [org_txt[i]]))
            i += 1
        else:
            doc_id = docs[i]
            k = -1
        k += 1
    return augm_txt


def main(args):
    org_text = open(args.f1, "r").read().splitlines()

    if args.f2 is None:
        context_text = org_text
    else:
        context_text = open(args.f2, "r").read().splitlines()

    doc_ids = open(args.doc_ids, "r").read().splitlines()

    if len(doc_ids) != len(org_text):
        raise Exception("There is a missmatch between the number of lines in the file and the document ids")

    # add contexts to the text file
    new_text = add_context(org_txt=org_text, context=context_text, docs=doc_ids, sep_token="</s>")

    out_name = args.f1 if args.name is None else args.name

    with open(out_name, "w") as fp:
        for line in new_text:
            fp.write(line)
            fp.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add context to the input file.')
    parser.add_argument('--f1', required=True, type=str, help='the original text')
    parser.add_argument('--doc_ids', required=True, type=str, help='the document ids or names of the segments')
    parser.add_argument('--f2', required=False, type=str, help='the text where the context will be taken from')
    parser.add_argument('--name', required=False, type=str, help='name of the final text')

    args = parser.parse_args()
    main(args)
