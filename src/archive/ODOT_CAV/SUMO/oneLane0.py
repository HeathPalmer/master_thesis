#!/usr/bin/env python

import os
import sys
import optparse
import time

#necessary to import xml2csv file from a different directory
#source:https://www.codegrepper.com/code-examples/python/import+script+from+another+folder+python
sys.path.append('/Program Files (x86)/Eclipse/Sumo/tools/xml')
import xml2csv

#used for writing xml files (better than examples)
import xml.etree.ElementTree as ET

# we need to import some python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


from sumolib import checkBinary  # Checks for the binary in environ vars
import traci


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options

# look up a new way to generate xml file.
# https://www.codegrepper.com/code-examples/python/python+string+to+xml
def generate_additionalfile(additionalFileName, inductionLoopFileName):
    #creating the rout file.
    with open(additionalFileName, "w") as additional:#os.path.join(subdirectory,"{}add.xml".format(recnum))
#         print("""<additional>
#         <additional>
#             <inductionLoop id="myLoop0" lane="e3_0" pos="10" freq="60" file="inductionLoop.xml" />
#             <inductionLoop id="myLoop1" lane="e3_1" pos="10" freq="60" file="indutionLoop1.xml" />
#             <inductionLoop id="myLoop2" lane="e3_2" pos="10" freq="60" file="inductionLoop2.xml" />
#         </additional>
# </additional>""", file=additional)
        print('<additional>', file=additional)
        print('\t<additional>', file=additional)  #\t used to indent in a print statement      
        print('\t\t<inductionLoop id="myLoop0" lane="e3_0" pos="10" freq="60" file="%s" />' % (inductionLoopFileName), file=additional)
        print('\t\t<inductionLoop id="myLoop1" lane="e3_1" pos="10" freq="60" file="%s" />' % (inductionLoopFileName), file=additional)
        print('\t\t<inductionLoop id="myLoop2" lane="e3_2" pos="10" freq="60" file="%s" />' % (inductionLoopFileName), file=additional)
        print('\t</additional>', file=additional)
        print('</additional>', file=additional)
# contains TraCI control loop
def run():
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        #print(step)

        #det_vehs = traci.inductionloop.getLastStepVehicleIDs("det_0")
        #for veh in det_vehs:
        #    print(veh)
        #    traci.vehicle.changeLane(veh, 2, 25)

        # if step == 100:
        #     traci.vehicle.changeTarget("1", "e9")
        #     traci.vehicle.changeTarget("3", "e9")

        step += 1

    traci.close()
    sys.stdout.flush()


# main entry point
if __name__ == "__main__":
    options = get_options()

    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    
    timestr = time.strftime("%Y%m%d")
    
    #create subdirectory or join it
    subdirectory = timestr + "_tripInfo"
    try:
        os.mkdir(subdirectory)
    except Exception:
        pass
    
    #set the file name based on increamenting value
    i = 0
    while os.path.exists(os.path.join(subdirectory,"%s_tripinfo.xml" % format(int(i), '03d'))):
        i += 1
    recnum = format(int(i), '03d')
    
    #generate route file
    additionalFileName = os.path.join(subdirectory,"{}.add.xml".format(recnum))
    inductionLoopFileName = "{}_induction.xml".format(recnum)
    generate_additionalfile(additionalFileName, inductionLoopFileName)
    
    
    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start([sumoBinary, "-c", "oneLane0.sumocfg", "--additional-files", additionalFileName,
                             "--collision.mingap-factor", "0",
                             "--tripinfo-output", os.path.join(subdirectory,"{}_tripinfo.xml".format(recnum)),
                             "--emission-output",os.path.join(subdirectory,"{}_emissions.xml".format(recnum))])
    run()
    #convert new xml file to csv
    tripInfoFileName = os.path.join(subdirectory,"{}_tripinfo.xml".format(recnum))
    emissionsInfoFileName = os.path.join(subdirectory,"{}_emissions.xml".format(recnum))
    inductionLoopFileName = os.path.join(subdirectory,"{}_induction.xml".format(recnum))
    xml2csv.main([tripInfoFileName])
    xml2csv.main([emissionsInfoFileName])
    xml2csv.main([inductionLoopFileName])
