import React, { Component } from 'react';
import { StyleSheet, Text, View, Image, TouchableOpacity } from 'react-native';

import Camera from '../components/Camera';

class Home extends Component {
  render() {
    return (
      <View style={styles.container}>
        <View style={styles.rect2StackStack}>
          <View style={styles.rect2Stack}>
            <Image
              source={require('../assets/images/blob1.png')}
              resizeMode='cover'
              style={styles.rect2}
            ></Image>
            <View style={styles.rect}>
              <Text style={styles.loremIpsum}>Hi! Welcome to Tomato-Living</Text>
              <View style={styles.rect6}>
                <Text style={styles.previousPictures}>What is Tomato-Living?</Text>

                <Text style={styles.infoTxt}>
                  ✔ Turn your Android phone into a mobile crop doctor.
                </Text>
                <Text style={styles.infoTxt}>
                  ✔ With just one photo, the app diagnoses infected tomato and
                  offers treatments for any disease.
                </Text>
                <Text style={styles.infoTxt}>
                  ✔ Benefit from agricultural experts' know-how or help fellow
                  farmers with your knowledge.
                </Text>
              </View>
              <View style={styles.rect4}>
                <Text style={styles.healYourCrop}>Heal Your Crop!</Text>
                <View style={styles.image3Row}>
                  <Image
                    source={require('../assets/images/qr.png')}
                    resizeMode='contain'
                    style={styles.image3}
                  ></Image>
                  <Image
                    source={require('../assets/images/next.png')}
                    resizeMode='contain'
                    style={styles.image6}
                  ></Image>
                  <Image
                    source={require('../assets/images/paper.png')}
                    resizeMode='contain'
                    style={styles.image4}
                  ></Image>
                  <Image
                    source={require('../assets/images/next.png')}
                    resizeMode='contain'
                    style={styles.image7}
                  ></Image>
                  <Image
                    source={require('../assets/images/healthcare-and-medical.png')}
                    resizeMode='contain'
                    style={styles.image5}
                  ></Image>
                </View>
                <Camera navigation={this.props.navigation} />
              </View>
            </View>
            <Image
              source={require('../assets/images/animation_500_kcit151v.gif')}
              resizeMode='contain'
              style={styles.image2}
            ></Image>
            <Image
              source={require('../assets/images/logo.png')}
              resizeMode='contain'
              style={styles.image10}
            ></Image>
          </View>
          <Image
            source={require('../assets/images/blob1.png')}
            resizeMode='cover'
            style={styles.rect3}
          ></Image>
        </View>
      </View>
    );
  }
}

export default Home;
// comicneuebold
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'white'
  },
  rect2: {
    top: -40,
    left: 110,
    height: 350,
    position: 'absolute'
  },

  rect: {
    top: 90,
    width: 278,
    height: 513,
    position: 'absolute',
    backgroundColor: '#ffffff',
    borderRadius: 35,
    shadowColor: 'rgba(0,0,0,1)',
    shadowOffset: {
      width: 0,
      height: 0
    },
    elevation: 7,
    shadowOpacity: 1,
    shadowRadius: 4,
    left: 28
  },
  infoTxt: {
    fontFamily: 'comicneuebold',
    color: '#195F57',
    top: 15,
    left: 10,
    width: 220,
    paddingTop: 5,
    textAlign: 'justify'
  },
  loremIpsum: {
    fontFamily: 'comicneuebold',
    color: '#195F57',
    fontSize: 20,
    marginTop: 19,
    marginLeft: 24
  },
  rect4: {
    width: 238,
    height: 176,
    backgroundColor: 'rgba(255,255,255,1)',
    borderRadius: 27,
    marginTop: 30,
    marginLeft: 20,
    shadowColor: 'rgba(0,0,0,1)',
    shadowOffset: {
      width: 1,
      height: 1
    },
    elevation: 5,
    shadowOpacity: 0.16,
    shadowRadius: 10
  },
  healYourCrop: {
    fontFamily: 'comicneuebold',
    color: '#195F57',
    fontSize: 18,
    marginTop: 14,
    marginLeft: 21
  },
  image3: {
    width: 38,
    height: 39,
    marginTop: 11
  },
  image6: {
    width: 16,
    height: 34,
    marginLeft: 12,
    marginTop: 14
  },
  image4: {
    width: 48,
    height: 48,
    marginLeft: 7,
    marginTop: 3
  },
  image7: {
    width: 16,
    height: 34,
    marginLeft: 1,
    marginTop: 13
  },
  image5: {
    width: 47,
    height: 53,
    marginLeft: 2
  },
  image3Row: {
    height: 53,
    flexDirection: 'row',
    marginTop: 27,
    marginLeft: 31,
    marginRight: 20
  },
  rect6: {
    width: 238,
    height: 205,
    backgroundColor: 'rgba(255,255,255,1)',
    borderRadius: 27,
    marginTop: 19,
    marginLeft: 22,
    shadowColor: 'rgba(0,0,0,1)',
    shadowOffset: {
      width: 1,
      height: 1
    },
    elevation: 5,
    shadowOpacity: 0.16,
    shadowRadius: 10
  },
  previousPictures: {
    fontFamily: 'comicneuebold',
    color: '#195F57',
    fontSize: 18,
    marginTop: 9,
    marginLeft: 14
  },
  rect8: {
    width: 35,
    height: 36,
    backgroundColor: '#E6E6E6',
    borderRadius: 5
  },
  rect10: {
    width: 35,
    height: 36,
    backgroundColor: '#E6E6E6',
    borderRadius: 5,
    marginLeft: 21
  },
  rect9: {
    width: 35,
    height: 36,
    backgroundColor: '#E6E6E6',
    borderRadius: 5,
    marginLeft: 23
  },
  rect11: {
    width: 35,
    height: 36,
    backgroundColor: '#E6E6E6',
    borderRadius: 5,
    marginLeft: 23
  },
  rect8Row: {
    height: 36,
    flexDirection: 'row',
    marginTop: 8,
    marginLeft: 17,
    marginRight: 14
  },
  rect7: {
    width: 238,
    height: 126,
    backgroundColor: 'rgba(255,255,255,1)',
    borderRadius: 27,
    marginTop: 17,
    marginLeft: 22,
    shadowColor: 'rgba(0,0,0,1)',
    shadowOffset: {
      width: 1,
      height: 1
    },
    elevation: 5,
    shadowOpacity: 0.16,
    shadowRadius: 10
  },
  today14Jul: {
    fontFamily: 'comicneuebold',
    color: '#195F57',
    fontSize: 14
  },
  today15: {
    fontFamily: 'comicneuebold',
    color: '#195F57',
    fontSize: 30,
    marginTop: 6
  },
  sunset632Pm: {
    fontFamily: 'comicneuebold',
    color: '#195F57',
    fontSize: 13,
    marginTop: 3,
    marginLeft: 1,
    width: 90
  },
  today14JulColumn: {
    width: 84,
    marginTop: 2
  },
  image9: {
    width: 78,
    height: 71,
    marginLeft: 46
  },
  today14JulColumnRow: {
    height: 72,
    flexDirection: 'row',
    marginTop: 14,
    marginLeft: 14,
    marginRight: 16
  },
  rainUntilAfternoon: {
    fontFamily: 'comicneuebold',
    color: '#195F57',
    fontSize: 13,
    marginTop: 18,
    marginLeft: 15
  },
  image2: {
    top: 492,
    left: 270,
    width: 131,
    height: 155,
    position: 'absolute'
  },
  sunset633: {
    top: 178,
    left: 64,
    position: 'absolute',
    fontFamily: 'comicneuebold',
    color: '#121212',
    fontSize: 13
  },
  image10: {
    top: 20,
    bottom: 20,
    left: -5,
    width: 120,
    height: 35,
    position: 'absolute'
  },
  rect2Stack: {
    top: 0,
    left: 57,
    width: 402,
    height: 647,
    position: 'absolute'
  },
  rect3: {
    top: 447,
    left: 0,
    width: 273,
    height: 245,
    position: 'absolute'
  },
  rect2StackStack: {
    width: 459,
    height: 692,
    marginLeft: -50
  }
});
