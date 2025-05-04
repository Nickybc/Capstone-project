"use client"

import { Clock, FileText, GitBranch } from "lucide-react"
import Link from "next/link"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ProjectTimeline } from "@/components/project-timeline"
import { TeamMembers } from "@/components/team-members"

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center space-x-4 sm:justify-between sm:space-x-0">
          <div className="flex gap-6 md:gap-10">
            <Link href="/" className="flex items-center space-x-2">
              <GitBranch className="h-6 w-6" />
              <span className="inline-block font-bold">Machine Learning-Based Credit Risk Evaluation</span>
            </Link>
          </div>
          <div className="flex flex-1 items-center justify-end space-x-4">
            <nav className="flex items-center space-x-1">
              <Button variant="ghost" size="sm">
                Overview
              </Button>
              <Button variant="ghost" size="sm">
                Progress
              </Button>
              <Button variant="ghost" size="sm">
                Team
              </Button>
              <Button variant="ghost" size="sm">
                Resources
              </Button>
            </nav>
          </div>
        </div>
      </header>
      <main className="flex-1">
        <section className="w-full py-12 md:py-24 lg:py-16 bg-slate-50 dark:bg-slate-900">
          <div className="container px-4 md:px-6">
            <div className="grid gap-6 lg:grid-cols-[1fr_400px] lg:gap-12 xl:grid-cols-[1fr_600px]">
              <div className="flex flex-col justify-center space-y-4">
                <div className="space-y-2">
                  <h1 className="text-3xl font-bold tracking-tighter sm:text-5xl xl:text-6xl/none">
                    Machine Learning-Based Credit Risk Evaluation
                  </h1>
                  <p className="max-w-[600px] text-gray-500 md:text-xl dark:text-gray-400">
                    An academic research project exploring advanced machine learning techniques for credit risk
                    assessment and prediction.
                  </p>
                </div>
                <div className="flex flex-col gap-2 min-[400px]:flex-row">
                  <Button
                    className="inline-flex h-10 items-center justify-center rounded-md bg-primary px-8 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50"
                    onClick={() => window.open("https://kdocs.cn/l/ccU5RkfLJ0h4", "_blank")}
                  >
                    View Documentation
                  </Button>
                  <Button
                    variant="outline"
                    className="inline-flex h-10 items-center justify-center rounded-md border border-input bg-background px-8 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50"
                    onClick={() => window.open("https://github.com/Nickybc/Capstone-project", "_blank")}
                  >
                    Project Repository
                  </Button>
                </div>
              </div>
              <div className="flex items-center justify-center">
                <Card className="w-full">
                  <CardHeader>
                    <CardTitle>Project Status</CardTitle>
                    <CardDescription>Current progress and upcoming milestones</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <div className="text-sm font-medium">Overall Progress</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">30%</div>
                        </div>
                        <Progress value={30} className="h-2" />
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <div className="text-sm font-medium">Data Collection</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">100%</div>
                        </div>
                        <Progress value={100} className="h-2" />
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <div className="text-sm font-medium">Feature Engineering</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">100%</div>
                        </div>
                        <Progress value={100} className="h-2" />
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <div className="text-sm font-medium">Model Development</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">20%</div>
                        </div>
                        <Progress value={20} className="h-2" />
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <div className="text-sm font-medium">Testing & Validation</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">10%</div>
                        </div>
                        <Progress value={10} className="h-2" />
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <div className="text-sm font-medium">Documentation</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">10%</div>
                        </div>
                        <Progress value={10} className="h-2" />
                      </div>
                    </div>
                  </CardContent>
                  <CardFooter>
                    <div className="flex items-center text-sm text-gray-500 dark:text-gray-400">
                      <Clock className="mr-1 h-4 w-4" />
                      Last updated: April 6, 2025
                    </div>
                  </CardFooter>
                </Card>
              </div>
            </div>
          </div>
        </section>
        <section className="w-full py-12 md:py-24 lg:py-16">
          <div className="container px-4 md:px-6">
            <div className="flex flex-col items-center justify-center space-y-4 text-center">
              <div className="space-y-2">
                <div className="inline-block rounded-lg bg-gray-100 px-3 py-1 text-sm dark:bg-gray-800">Team</div>
                <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">Meet Our Research Team</h2>
                <p className="max-w-[900px] text-gray-500 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed dark:text-gray-400">
                  Our dedicated team of researchers working on the Machine Learning-Based Credit Risk Evaluation
                  project.
                </p>
              </div>
            </div>
            <TeamMembers />
          </div>
        </section>
        <section className="w-full py-12 md:py-24 lg:py-16 bg-slate-50 dark:bg-slate-900">
          <div className="container px-4 md:px-6">
            <div className="flex flex-col items-center justify-center space-y-4 text-center">
              <div className="space-y-2">
                <div className="inline-block rounded-lg bg-gray-100 px-3 py-1 text-sm dark:bg-gray-800">Timeline</div>
                <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
                  Project Assessment Timeline
                </h2>
                <p className="max-w-[900px] text-gray-500 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed dark:text-gray-400">
                  Official assessment milestones and deadlines for the 2024-25 academic year project.
                </p>
                <div className="mt-4 flex items-center justify-center gap-6 text-sm">
                  <div className="flex items-center">
                    <div className="mr-2 h-3 w-3 rounded-full bg-green-500"></div>
                    <span>Completed</span>
                  </div>
                  <div className="flex items-center">
                    <div className="mr-2 h-3 w-3 rounded-full bg-yellow-500"></div>
                    <span>In Progress</span>
                  </div>
                  <div className="flex items-center">
                    <div className="mr-2 h-3 w-3 rounded-full bg-gray-300 dark:bg-gray-600"></div>
                    <span>Upcoming</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="mx-auto grid max-w-5xl items-center gap-6 py-12 lg:grid-cols-1 lg:gap-12">
              <ProjectTimeline />
            </div>
          </div>
        </section>
        <section className="w-full py-12 md:py-24 lg:py-16">
          <div className="container px-4 md:px-6">
            <Tabs defaultValue="overview" className="w-full">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="methodology">Methodology</TabsTrigger>
                <TabsTrigger value="results">Results</TabsTrigger>
                <TabsTrigger value="resources">Resources</TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Project Overview</CardTitle>
                    <CardDescription>Machine Learning-Based Credit Risk Evaluation</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <p>
                      With the increasing demand for financial services, effective credit risk assessment has become
                      essential for lending institutions. This project aims to develop a comprehensive and robust credit
                      risk assessment model to accurately predict loan repayment probabilities, reduce fraud rates, and
                      significantly limit the occurrence of bad debts.
                    </p>

                    <h3 className="text-lg font-semibold">Methodology:</h3>
                    <p>
                      The core methodology will primarily leverage advanced deep learning techniques, notably Long
                      Short-Term Memory (LSTM) networks, due to their proficiency in capturing sequential patterns in
                      credit-related data. Traditional machine learning approaches, including Linear Regression, Random
                      Forest, and XGBoost, will be used as comparative benchmarks to highlight the advantages and
                      improvements brought by deep learning.
                    </p>

                    <h3 className="text-lg font-semibold">Model Interpretability:</h3>
                    <p>
                      Model interpretability is another key objective, aiming to provide internal stakeholders such as
                      credit analysts and risk managers with clear and actionable insights. This will empower them to
                      make informed decisions, particularly when issuing financial products such as credit cards.
                      Additionally, the interpretability component is intended to benefit end customers by offering
                      transparent guidance on ways they can enhance their creditworthiness.
                    </p>

                    <h3 className="text-lg font-semibold">Deployment:</h3>
                    <p>
                      The final model will be deployed through a user-friendly API, facilitating seamless integration
                      within existing infrastructures. Moreover, an interactive web-based interface created with
                      Streamlit will be provided, enabling real-time credit evaluations and enhancing the
                      decision-making process for both internal stakeholders and end customers.
                    </p>

                    <h3 className="text-lg font-semibold">Research Objectives:</h3>
                    <ul className="list-disc pl-6 space-y-2">
                      <li>Develop and compare multiple machine learning models for credit risk prediction</li>
                      <li>Implement LSTM networks for capturing sequential patterns in credit data</li>
                      <li>Create interpretable models that provide insights into decision-making processes</li>
                      <li>Evaluate model performance against traditional credit scoring methods</li>
                      <li>Deploy a user-friendly API and web interface for real-time credit evaluations</li>
                    </ul>
                  </CardContent>
                </Card>
              </TabsContent>
              <TabsContent value="methodology" className="mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Research Methodology</CardTitle>
                    <CardDescription>Our approach to developing ML-based credit risk models</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <h3 className="text-lg font-semibold">Data Collection and Preprocessing</h3>
                    <p>
                      We've collected financial data from multiple sources, including credit histories, transaction
                      records, and demographic information. The data preprocessing phase involved handling missing
                      values, feature engineering, and normalization.
                    </p>

                    <h3 className="text-lg font-semibold">Model Development</h3>
                    <p>We're implementing and comparing several machine learning algorithms:</p>
                    <ul className="list-disc pl-6 space-y-2">
                      <li>Deep Learning: Long Short-Term Memory (LSTM) networks</li>
                      <li>Linear Regression (baseline model)</li>
                      <li>Random Forest</li>
                      <li>XGBoost</li>
                      <li>Ensemble methods</li>
                    </ul>

                    <h3 className="text-lg font-semibold">Model Interpretability</h3>
                    <p>We're focusing on creating interpretable models using techniques such as:</p>
                    <ul className="list-disc pl-6 space-y-2">
                      <li>SHAP (SHapley Additive exPlanations) values</li>
                      <li>Feature importance analysis</li>
                      <li>Partial dependence plots</li>
                      <li>Rule extraction from complex models</li>
                    </ul>

                    <h3 className="text-lg font-semibold">Evaluation Metrics</h3>
                    <p>Model performance is being evaluated using:</p>
                    <ul className="list-disc pl-6 space-y-2">
                      <li>Area Under the ROC Curve (AUC)</li>
                      <li>Precision, Recall, and F1 Score</li>
                      <li>Confusion Matrix</li>
                      <li>Kolmogorov-Smirnov statistic</li>
                      <li>Expected monetary value analysis</li>
                    </ul>

                    <h3 className="text-lg font-semibold">Deployment Strategy</h3>
                    <p>The final model will be deployed through:</p>
                    <ul className="list-disc pl-6 space-y-2">
                      <li>RESTful API for system integration</li>
                      <li>Interactive Streamlit web interface</li>
                      <li>Comprehensive documentation for stakeholders</li>
                    </ul>
                  </CardContent>
                </Card>
              </TabsContent>
              <TabsContent value="results" className="mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Preliminary Results</CardTitle>
                    <CardDescription>Current findings and ongoing analysis</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <p>Our research is still in progress, but we have some preliminary results to share:</p>

                    <h3 className="text-lg font-semibold">Model Performance Comparison</h3>
                    <p>
                      Initial testing shows that ensemble methods outperform individual models, with Gradient Boosting
                      achieving the highest AUC score of 0.89.
                    </p>

                    <h3 className="text-lg font-semibold">Feature Importance</h3>
                    <p>We've identified several key features that strongly correlate with credit risk:</p>
                    <ul className="list-disc pl-6 space-y-2">
                      <li>Payment history (most significant)</li>
                      <li>Debt-to-income ratio</li>
                      <li>Length of credit history</li>
                      <li>Recent credit inquiries</li>
                      <li>Types of credit in use</li>
                    </ul>

                    <h3 className="text-lg font-semibold">Next Steps</h3>
                    <p>We are currently working on:</p>
                    <ul className="list-disc pl-6 space-y-2">
                      <li>Fine-tuning model hyperparameters</li>
                      <li>Implementing explainable AI techniques</li>
                      <li>Conducting cross-validation tests</li>
                      <li>Preparing a comprehensive evaluation report</li>
                    </ul>
                  </CardContent>
                </Card>
              </TabsContent>
              <TabsContent value="resources" className="mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Project Resources</CardTitle>
                    <CardDescription>Documentation, code, and research materials</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid gap-4 md:grid-cols-2">
                      <Card>
                        <CardHeader className="p-4">
                          <CardTitle className="text-base">Project Documentation</CardTitle>
                        </CardHeader>
                        <CardContent className="p-4 pt-0">
                          <ul className="space-y-2">
                            <li className="flex items-center">
                              <FileText className="mr-2 h-4 w-4" />
                              <Link href="#" className="text-blue-600 hover:underline">
                                Research Proposal
                              </Link>
                            </li>
                            <li className="flex items-center">
                              <FileText className="mr-2 h-4 w-4" />
                              <Link href="#" className="text-blue-600 hover:underline">
                                Literature Review
                              </Link>
                            </li>
                            <li className="flex items-center">
                              <FileText className="mr-2 h-4 w-4" />
                              <Link href="#" className="text-blue-600 hover:underline">
                                Methodology Document
                              </Link>
                            </li>
                            <li className="flex items-center">
                              <FileText className="mr-2 h-4 w-4" />
                              <Link href="#" className="text-blue-600 hover:underline">
                                Progress Reports
                              </Link>
                            </li>
                          </ul>
                        </CardContent>
                      </Card>
                      <Card>
                        <CardHeader className="p-4">
                          <CardTitle className="text-base">Code & Data</CardTitle>
                        </CardHeader>
                        <CardContent className="p-4 pt-0">
                          <ul className="space-y-2">
                            <li className="flex items-center">
                              <GitBranch className="mr-2 h-4 w-4" />
                              <Link
                                href="https://github.com/Nickybc/Capstone-project"
                                target="_blank"
                                className="text-blue-600 hover:underline"
                              >
                                GitHub Repository
                              </Link>
                            </li>
                            <li className="flex items-center">
                              <FileText className="mr-2 h-4 w-4" />
                              <Link href="#" className="text-blue-600 hover:underline">
                                Dataset Documentation
                              </Link>
                            </li>
                            <li className="flex items-center">
                              <FileText className="mr-2 h-4 w-4" />
                              <Link href="#" className="text-blue-600 hover:underline">
                                Model Implementations
                              </Link>
                            </li>
                            <li className="flex items-center">
                              <FileText className="mr-2 h-4 w-4" />
                              <Link href="#" className="text-blue-600 hover:underline">
                                Evaluation Scripts
                              </Link>
                            </li>
                          </ul>
                        </CardContent>
                      </Card>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        </section>
      </main>
      <footer className="w-full border-t py-6 md:py-0">
        <div className="container flex flex-col items-center justify-between gap-4 md:h-24 md:flex-row">
          <p className="text-center text-sm leading-loose text-gray-500 md:text-left">
            © 2025 Machine Learning-Based Credit Risk Evaluation. All rights reserved.
          </p>
          <div className="flex items-center gap-4">
            <Link href="#" className="text-sm text-gray-500 hover:underline">
              Privacy Policy
            </Link>
            <Link href="#" className="text-sm text-gray-500 hover:underline">
              Terms of Service
            </Link>
            <Link href="#" className="text-sm text-gray-500 hover:underline">
              Contact
            </Link>
          </div>
        </div>
      </footer>
    </div>
  )
}
