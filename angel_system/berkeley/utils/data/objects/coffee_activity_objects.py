sub_steps = {}
sub_steps['step 1'] = []
sub_steps['step 2'] = []
sub_steps['step 3'] = []
sub_steps['step 4'] = []
sub_steps['step 5'] = []
sub_steps['step 6'] = []
sub_steps['step 7'] = []
sub_steps['step 8'] = []

###################################################step 1###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('Measure 12 ounces of cold water')
sub_step.append([['measuring cup', 'water']])
sub_steps['step 1'].append(sub_step)
del sub_step

##############################sub-step 2##############################
sub_step = []
sub_step.append('Transfer the water to a kettle')
sub_step.append([['measuring cup', 'kettle (open)']])
sub_steps['step 1'].append(sub_step)
del sub_step

##############################sub-step 3##############################
sub_step = []
sub_step.append('Turn on the kettle')
sub_step.append([['switch', 'hand']])
sub_steps['step 1'].append(sub_step)
del sub_step

###################################################step 2###################################################
##############################sub-step 4##############################
sub_step = []
sub_step.append('Place the Dripper on top of the mug')
sub_step.append([['hand', 'filter cone'],
                    ['hand', 'mug'],
                    ['mug', 'filter cone'],
                    ['hand', 'filter cone + mug']])
sub_steps['step 2'].append(sub_step)
del sub_step

###################################################step 3###################################################
#############################sub-step 5##############################
sub_step = []
sub_step.append('Take the coffee filter and fold it in half into a semicircle.')
sub_step.append([['paper filter', 'paper filter bag'],
    ['hand', 'paper filter bag'],
    ['hand', 'paper filter'],
    ['hand', 'paper filter (semi)']])
sub_steps['step 3'].append(sub_step)
del sub_step

##############################sub-step 6##############################
sub_step = []
sub_step.append('Fold the coffee filter again to create a quarter circle.')
sub_step.append([['hand', 'paper filter (quarter)']])
sub_steps['step 3'].append(sub_step)
del sub_step

# #############################sub-step 7##############################
sub_step = []
sub_step.append('Place the folder paper into the dripper.')
sub_step.append([['paper filter (quarter)', 'filter cone + mug']])
sub_steps['step 3'].append(sub_step)
del sub_step

##############################sub-step 8##############################
sub_step = []
sub_step.append('Spread the filter open to create a cone inside the dripper.')
sub_step.append([['hand', 'paper filter + filter cone + mug']])
sub_steps['step 3'].append(sub_step)
del sub_step

###################################################step 4###################################################
##############################sub-step 9##############################
sub_step = []
sub_step.append('Turn on the kitchen scale.')
sub_step.append([['hand', 'scale (off)'],
                    ['hand', 'scale (on)']])
sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 10##############################
sub_step = []
sub_step.append('Place a bowl on the scale.')
sub_step.append([['container', 'scale (off)'],
                    ['container', 'scale (on)']])
sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 11##############################
sub_step = []
sub_step.append('Zero out the kitchen scale.')
sub_step.append([['hand', 'container + scale']])
sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 12##############################
sub_step = []
sub_step.append('Add coffee beans into the bowl until read 25 grams.')
sub_step.append([['container + scale', 'coffee bag'],
                    ['coffee beans + container + scale', 'coffee bag'],
                    ['hand', 'coffee bag']])
sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 13##############################
sub_step = []
sub_step.append('Poured the measured beans into the coffee grinder.')
sub_step.append([['coffee beans + container', 'grinder (open)'],
                    ['container', 'grinder (open)']])
sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 14##############################
sub_step = []
sub_step.append('Use timer')
sub_step.append([['hand', 'timer (else)'],
                    ['hand', 'timer (20)'],
                    ['hand', 'timer (30)']])
sub_steps['step 4'].append(sub_step)
del sub_step

# ##############################sub-step 15##############################
sub_step = []
sub_step.append('Grind the coffee beans by pressing and holding down the back part')
sub_step.append([['hand', 'grinder (close)']])
sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 16##############################
sub_step = []
sub_step.append('Pour the grinded coffee beans into the filter cone.')
sub_step.append([['paper filter + filter cone + mug', 'grinder (open)'],
                    ['paper filter + filter cone', 'grinder (open)'],
                    ['coffee grounds + paper filter + filter cone + mug', 'grinder (open)'],
                    ['coffee grounds + paper filter + filter cone', 'grinder (open)']])
sub_steps['step 4'].append(sub_step)
del sub_step

###################################################step 5###################################################
##############################sub-step 17##############################
sub_step = []
sub_step.append('Turn on the thermometer.')
sub_step.append([['thermometer (open)', 'hand'],
                    ['thermometer (close)', 'hand']])
sub_steps['step 5'].append(sub_step)
del sub_step

##############################sub-step 18##############################
sub_step = []
sub_step.append('Place the end of the thermometer into the water.')
sub_step.append([['thermometer (open)', 'kettle (open)']])
sub_steps['step 5'].append(sub_step)
del sub_step

###################################################step 6###################################################
##############################sub-step 18##############################
# sub_step = []
# sub_step.append('Set timer to 30 seconds.')
# sub_step.append([['hand', 'timer'],
#                  ['hand', 'timer (20)'],
#                  ['hand', 'timer (0)'],
#                  ['hand', 'timer (30)']])
# sub_steps['step 6'].append(sub_step)
# del sub_step

##############################sub-step 19##############################
sub_step = []
sub_step.append('Pour small amount of water onto the grounds.')
sub_step.append([['kettle', 'coffee grounds + paper filter + filter cone + mug']])
sub_steps['step 6'].append(sub_step)
del sub_step

##############################sub-step 20##############################
# sub_step = []
# sub_step.append('Wait about 30 seconds.')
# sub_step.append([
#     # ['hand', 'timer'],
#     #              ['hand', 'timer (20)'],
#     #              ['hand', 'timer (0)'],
#     #              ['hand', 'timer (30)']
# ])
# sub_steps['step 6'].append(sub_step)
# del sub_step

###################################################step 7###################################################
#############################sub-step 20##############################
sub_step = []
sub_step.append(
    'Slowly pour water into the grounds circular motion.')
sub_step.append([['kettle', 'water + coffee grounds + paper filter + filter cone + mug']])
sub_steps['step 7'].append(sub_step)
del sub_step

##################################################step 8###################################################
#############################sub-step 21##############################
sub_step = []
sub_step.append('Remove dripper from cup.')
sub_step.append([['hand', 'used paper filter + filter cone'],
                    ['hand', 'used paper filter + filter cone + mug']])
sub_steps['step 8'].append(sub_step)

##############################sub-step 22##############################
sub_step = []
sub_step.append('Remove the coffee grounds and paper filter from the dripper.')
sub_step.append([
                # ['hand', 'used paper filter + filter cone'],
                    # ['filter cone', 'used paper filter'],
                    ['hand', 'used paper filter']])
sub_steps['step 8'].append(sub_step)

#############################sub-step 23##############################
sub_step = []
sub_step.append('Discard the coffee grounds and paper filter.')
sub_step.append([['used paper filter', 'trash can']
                    # , ['hand', 'used paper filter']
                    ])
sub_steps['step 8'].append(sub_step)


# This version of the sub-steps allows us to make 1:1 matches with the activity labels
original_sub_steps = {}
original_sub_steps['step 1'] = []
original_sub_steps['step 2'] = []
original_sub_steps['step 3'] = []
original_sub_steps['step 4'] = []
original_sub_steps['step 5'] = []
original_sub_steps['step 6'] = []
original_sub_steps['step 7'] = []
original_sub_steps['step 8'] = []

###################################################step 1###################################################
##############################sub-step 1##############################
sub_step = []
sub_step.append('Measure 12 ounces of water in the liquid measuring cup')
sub_step.append([['measuring cup', 'water']])
original_sub_steps['step 1'].append(sub_step)
del sub_step

##############################sub-step 2##############################
sub_step = []
sub_step.append('Pour the water from the liquid measuring cup into the electric kettle')
sub_step.append([['measuring cup', 'kettle (open)']])
original_sub_steps['step 1'].append(sub_step)
del sub_step

##############################sub-step 3##############################
sub_step = []
sub_step.append('Turn on the kettle')
sub_step.append([['switch', 'hand']])
original_sub_steps['step 1'].append(sub_step)
del sub_step

###################################################step 2###################################################
##############################sub-step 4##############################
sub_step = []
sub_step.append('Place the Dripper on top of the mug')
sub_step.append([['hand', 'filter cone'],
                    ['hand', 'mug'],
                    ['mug', 'filter cone'],
                    ['hand', 'filter cone + mug']])
original_sub_steps['step 2'].append(sub_step)
del sub_step

###################################################step 3###################################################
#############################sub-step 5##############################
sub_step = []
sub_step.append('Take the coffee filter and fold it in half to create a semi-circle')
sub_step.append([['paper filter', 'paper filter bag'],
    ['hand', 'paper filter bag'],
    ['hand', 'paper filter'],
    ['hand', 'paper filter (semi)']])
original_sub_steps['step 3'].append(sub_step)
del sub_step

##############################sub-step 6##############################
sub_step = []
sub_step.append('Fold the filter in half again to create a quarter-circle')
sub_step.append([['hand', 'paper filter (quarter)']])
original_sub_steps['step 3'].append(sub_step)
del sub_step

# #############################sub-step 7##############################
sub_step = []
sub_step.append('Place the folded filter into the dripper such that the the point of the quarter-circle rests in the center of the dripper')
sub_step.append([['paper filter (quarter)', 'filter cone + mug']])
original_sub_steps['step 3'].append(sub_step)
del sub_step

##############################sub-step 8##############################
sub_step = []
sub_step.append('Spread the filter open to create a cone inside the dripper.')
sub_step.append([['hand', 'paper filter + filter cone + mug']])
original_sub_steps['step 3'].append(sub_step)
del sub_step

###################################################step 4###################################################
##############################sub-step 9##############################
sub_step = []
sub_step.append('Turn on the kitchen scale')
sub_step.append([['hand', 'scale (off)'],
                    ['hand', 'scale (on)']])
original_sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 10##############################
sub_step = []
sub_step.append('Place a bowl on the scale')
sub_step.append([['container', 'scale (off)'],
                    ['container', 'scale (on)']])
original_sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 11##############################
sub_step = []
sub_step.append('Zero the scale')
sub_step.append([['hand', 'container + scale']])
original_sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 12##############################
sub_step = []
sub_step.append('Add coffee beans to the bowl until the scale reads 25 grams')
sub_step.append([['container + scale', 'coffee bag'],
                    ['coffee beans + container + scale', 'coffee bag'],
                    ['hand', 'coffee bag']])
original_sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 13##############################
sub_step = []
sub_step.append('Pour the measured coffee beans into the coffee grinder')
sub_step.append([['coffee beans + container', 'grinder (open)'],
                    ['container', 'grinder (open)']])
original_sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 14##############################
sub_step = []
sub_step.append('Set timer for 20 seconds')
sub_step.append([['hand', 'timer'],
                 ['hand', 'timer (20)'],
                 ['hand', 'timer (else)']])
original_sub_steps['step 4'].append(sub_step)
del sub_step

sub_step = []
sub_step.append('Set timer to 30 seconds')
sub_step.append([['hand', 'timer'],
                 ['hand', 'timer (else)'],
                 ['hand', 'timer (30)']])
original_sub_steps['step 4'].append(sub_step)
del sub_step

sub_step = []
sub_step.append('Turn on the timer')
sub_step.append([['hand', 'timer'],
                 ['hand', 'timer (else)']])
original_sub_steps['step 4'].append(sub_step)
del sub_step

# ##############################sub-step 15##############################
sub_step = []
sub_step.append('Grind the coffee beans by pressing and holding down on the black part of the lid')
sub_step.append([['hand', 'grinder (close)']])
original_sub_steps['step 4'].append(sub_step)
del sub_step

##############################sub-step 16##############################
sub_step = []
sub_step.append('Pour the grounded coffee beans into the filter cone prepared in step 2')
sub_step.append([['paper filter + filter cone + mug', 'grinder (open)'],
                    ['paper filter + filter cone', 'grinder (open)'],
                    ['coffee grounds + paper filter + filter cone + mug', 'grinder (open)'],
                    ['coffee grounds + paper filter + filter cone', 'grinder (open)']])
original_sub_steps['step 4'].append(sub_step)
del sub_step

###################################################step 5###################################################
##############################sub-step 17##############################
sub_step = []
sub_step.append('Turn on the thermometer')
sub_step.append([['thermometer (open)', 'hand'],
                    ['thermometer (close)', 'hand']])
original_sub_steps['step 5'].append(sub_step)
del sub_step

##############################sub-step 18##############################
sub_step = []
sub_step.append('Place the end of the thermometer into the water')
sub_step.append([['thermometer (open)', 'kettle (open)']])
original_sub_steps['step 5'].append(sub_step)
del sub_step

###################################################step 6###################################################
##############################sub-step 18##############################

##############################sub-step 19##############################
sub_step = []
sub_step.append('Pour a small amount of water over the grounds in order to wet the grounds')
sub_step.append([['kettle', 'coffee grounds + paper filter + filter cone + mug']])
original_sub_steps['step 6'].append(sub_step)
del sub_step

###################################################step 7###################################################
#############################sub-step 20##############################
sub_step = []
sub_step.append('Slowly pour the water over the grounds in a circular motion. Do not overfill beyond the top of the paper filter')
sub_step.append([['kettle', 'water + coffee grounds + paper filter + filter cone + mug']])
original_sub_steps['step 7'].append(sub_step)
del sub_step

# ???? Allow the rest of the water in the dripper to drain

##################################################step 8###################################################
#############################sub-step 21##############################
sub_step = []
sub_step.append('Remove the dripper from the cup')
sub_step.append([['hand', 'used paper filter + filter cone'],
                    ['hand', 'used paper filter + filter cone + mug']])
original_sub_steps['step 8'].append(sub_step)

##############################sub-step 22##############################
sub_step = []
sub_step.append('Remove the coffee grounds and paper filter from the dripper')
sub_step.append([['hand', 'used paper filter + filter cone'],
                 ['hand', 'used paper filter']])
original_sub_steps['step 8'].append(sub_step)

#############################sub-step 23##############################
sub_step = []
sub_step.append('Discard the coffee grounds and paper filter')
sub_step.append([['used paper filter', 'trash can'],
                 ['hand', 'used paper filter']
                    ])
original_sub_steps['step 8'].append(sub_step)





contact_pairs_details = [[['measuring cup', 'water'],
         ['measuring cup', 'kettle (open)'],
         ['switch', 'hand']],  # step 1

         [['hand', 'filter cone'],
          ['mug', 'filter cone'],
          ['hand', 'filter cone + mug']],  # step 2

         [['paper filter', 'paper filter bag'],
          ['hand', 'paper filter bag'],
          ['hand', 'paper filter'],
          ['hand', 'paper filter (semi)'],
          ['hand', 'paper filter (quarter)'],
          ['paper filter (quarter)', 'filter cone + mug'],
          ['hand', 'paper filter + filter cone + mug']], # step 3

         [['hand', 'scale (on)'],
          ['hand', 'scale (off)'],
          ['hand', 'container + scale'],
          ['scale (on)', 'container'],
          ['scale (off)', 'container'],
          ['container + scale', 'coffee bag'],
          ['coffee beans + container + scale', 'coffee bag'],
          ['coffee beans + container', 'grinder (open)'],
          ['container', 'grinder (open)'],
          ['hand', 'timer (else)'],
          ['hand', 'timer (30)'],
          ['hand', 'timer (20)'],
          ['hand', 'grinder (close)'],
          ['paper filter + filter cone + mug', 'grinder (open)'],
          ['paper filter + filter cone', 'grinder (open)'],
          ['coffee grounds + paper filter + filter cone + mug', 'grinder (open)'],
          ['coffee grounds + paper filter + filter cone', 'grinder (open)']],  # step 4

         [['thermometer (open)', 'kettle (open)'],
          ['thermometer (open)', 'hand'],
         ['thermometer (close)', 'hand']], # step 5

         [['kettle', 'coffee grounds + paper filter + filter cone + mug'],
          # ['kettle', 'water + coffee grounds + paper filter + filter cone + mug'],
          # ['hand', 'timer']
          ], # step 6

        [['kettle', 'water + coffee grounds + paper filter + filter cone + mug']
         # ['kettle', 'used paper filter + filter cone + mug']
         ],  # step 7

         [['mug', 'used paper filter + filter cone'],
          ['hand', 'used paper filter + filter cone'],
          ['hand', 'used paper filter + filter cone + mug'],
          ['used paper filter', 'trash can'],
          ['trash can', 'filter cone'],
          ['hand', 'used paper filter']]  # step 8
         ]



States_Pairs = [['kettle',
                    'kettle (open)'],
                ['coffee beans + container', 'coffee beans + container + scale'],
                ['coffee grounds + paper filter + filter cone',
                    'coffee grounds + paper filter + filter cone + mug',
                    'filter cone', 'filter cone + mug', 'paper filter + filter cone',
                    'paper filter + filter cone + mug', 'used paper filter + filter cone',
                    'used paper filter + filter cone + mug', 'water + coffee grounds + paper filter + filter cone + mug'],
                ['coffee + mug',
                    'coffee grounds + paper filter + filter cone + mug',
                    'filter cone + mug', 'mug', 'paper filter + filter cone + mug',
                    'used paper filter + filter cone + mug', 'water + coffee grounds + paper filter + filter cone + mug'],
                ['container', 'container + scale'],
                ['scale (off)', 'scale (on)', 'container + scale', 'coffee beans + container + scale'],
                ['paper filter (semi)', 'paper filter (quarter)', 'paper filter'],
                ['coffee beans + container', 'coffee beans + container + scale'],
                ['timer (else)', 'timer (20)', 'timer (30)'],
                ['thermometer (open)', 'thermometer (close)'],
                ['grinder (close)', 'grinder (open)']
                ]



CONTACT_PAIRS_v1 = [['measuring cup (empty)', 'water'],
                         ['measuring cup (full)', 'water'],
                         ['measuring cup (full)', 'kettle (full)'],
                         ['measuring cup (full)', 'kettle (empty)'],
                         ['measuring cup (empty)', 'kettle (full)'],
                         ['measuring cup (empty)', 'kettle (empty)'],  # step 1

                         ['mug', 'filter cone'], # step 2

                         ['paper filter', 'paper filter bag'],
                         ['paper filter (semi)', 'filter cone + mug'],
                         ['paper filter (quarter)', 'filter cone + mug'],
                         ['paper filter', 'filter cone + mug'], # step 3

                         ['scale (on)', 'container'],
                         ['scale (off)', 'container'],
                         ['container + scale', 'coffee bag'],
                         ['coffee beans + container + scale', 'coffee bag'],
                         ['coffee beans + container', 'grinder'],
                         ['container', 'grinder'],
                         ['paper filter + filter cone + mug', 'grinder'],
                         ['paper filter + filter cone', 'grinder'],
                         ['coffee beans + paper filter + filter cone + mug', 'grinder'],
                         ['coffee beans + paper filter + filter cone', 'grinder'],
                         ['coffee grounds + paper filter + filter cone', 'grinder'],
                         ['coffee grounds + paper filter + filter cone + mug', 'grinder'],# step 4

                         ['thermometer', 'kettle (full)'],
                         ['thermometer', 'kettle (empty)'],
                         # ['thermometer', 'kettle'], # step 5

                         ['kettle', 'coffee grounds + paper filter + filter cone + mug'],
                         ['kettle', 'water + coffee grounds + paper filter + filter cone + mug'],
                         ['kettle', 'used paper filter + filter cone + mug'],  # step 6 ~ 7

                         ['mug', 'used paper filter + filter cone'],
                         ['used paper filter', 'filter cone'],
                         ['used paper filter + filter cone', 'paper towel'],
                         ['used paper filter', 'trash can'],
                         ['trash can', 'filter cone']# step 8
                         ]
                         
CONTACT_PAIRS = [['measuring cup', 'water'],
                    ['kettle (open)', 'measuring cup'],# step 1

                    ['mug', 'filter cone'],  # step 2

                    ['paper filter', 'paper filter bag'],
                    ['paper filter (semi)', 'filter cone + mug'],
                    ['paper filter (quarter)', 'filter cone + mug'],
                    ['paper filter', 'filter cone + mug'],  # step 3

                    ['scale (on)', 'container'],
                    ['scale (off)', 'container'],
                    ['container + scale', 'coffee bag'],
                    ['coffee beans + container + scale', 'coffee bag'],
                    ['coffee beans + container', 'grinder (open)'],
                    ['container', 'grinder (open)'],
                    ['paper filter + filter cone + mug', 'grinder (open)'],
                    ['paper filter + filter cone', 'grinder (open)'],
                    ['coffee grounds + paper filter + filter cone', 'grinder (open)'],
                    ['coffee grounds + paper filter + filter cone + mug', 'grinder (open)'],  # step 4

                    ['thermometer (open)', 'kettle (open)'],
                    # ['thermometer', 'kettle'], # step 5

                    ['kettle', 'coffee grounds + paper filter + filter cone + mug'],
                    ['kettle', 'water + coffee grounds + paper filter + filter cone + mug'],  # step 6 ~ 7

                    ['mug', 'used paper filter + filter cone'],
                    # ['hand', 'used paper filter + filter cone'],
                    # ['hand', 'used paper filter + filter cone + mug'],
                    ['used paper filter', 'trash can'],
                    ['trash can', 'filter cone'],
                    ['hand', 'used paper filter']  # step 8
                    ]

