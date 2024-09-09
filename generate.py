#!/usr/bin/env python3

import argparse
import joblib

from common import GetLabels, LABELS

LIB_EMLEARN = "emlearn"
LIB_M2CGEN  = "m2cgen"
LIB_MICROML = "micromlgen"

if __name__ == "__main__":
    prs = argparse.ArgumentParser()
    prs.add_argument("-l", "--library", type=str,
                     choices=[LIB_EMLEARN, LIB_M2CGEN, LIB_MICROML],
                     default=LIB_EMLEARN,
                     help="Select the ML library to use")
    prs.add_argument("-i", "--input", type=str, default="model", help="Specify the path of the model")
    prs.add_argument("-o", "--output", type=str, default="model.h", help="Output filename for the C model")
    prs.add_argument("-c", "--class-count", type=int, default=len(LABELS), help="Specify the number of classes for classification")
    prs.add_argument("--no-convert", action="store_true", help="Do not convert from C++ to C when using micromlgen")
    args = prs.parse_args()

    model = joblib.load(args.input)
    if args.library == LIB_EMLEARN:
        import emlearn
        from emlearn.evaluate import trees

        modelSize = emlearn.evaluate.trees.model_size_bytes(model)
        modelNodeCount = emlearn.evaluate.trees.model_size_nodes(model)

        print("Model size: {} bytes".format(modelSize))
        print("Node count: {}".format(modelNodeCount))

        cmodel = emlearn.convert(model)
        cmodel.save(file=args.output)
    elif args.library == LIB_M2CGEN:
        import m2cgen as m2c

        code = m2c.export_to_c(model)
        with open(args.output, 'w') as f:
            f.write(code)
    else:
        from micromlgen import port

        with open(args.output, 'w') as f:
            content = port(model)
            if not args.no_convert:
                # Remove the unnecessary C++, which makes compiling for RIOT more troublesome
                # for no reason at all.

                #print(content[-100:])
                #print(content[:300])
                startOffset = 0
                for i in range(0, len(content) - 3):
                    if content[i:i+3] == "int":
                        startOffset = i
                        break

                endOffset = len(content) - 1
                for i in range(endOffset, 0, -1):
                    if content[i] == ':':
                        if content[i-9:i] == "protected":
                            endOffset = i - 9
                            break
                content = content[startOffset:endOffset]

            f.write(content)

    print("Exported model '{}' to '{}'".format(args.input, args.output))

    # Export the list of labels for the C program
    content = ""
    for lb in GetLabels(args.class_count):
        content += '"{}",\n'.format(lb)

    with open("labels.h", 'w') as f:
        f.write(content)
    print("Wrote labels.h")
